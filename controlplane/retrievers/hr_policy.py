"""
HR Policy KB adapter.

Vector side: the existing parent-child FAISS - MiniLM finds CHILD chunks, we
resolve each to its unique PARENT section (full context) via parent_store.json.

BM25 side: the repo ships no BM25 for HR policy, so `scripts/build_hr_bm25.py`
builds `bm25_parents.pkl` = {"bm25": BM25Okapi, "parent_ids": [...]} over the
parent sections. If that file is missing we build an in-memory BM25 on first load
(cheap - a few hundred short sections).
"""

from __future__ import annotations

import json
import pickle
import re
from typing import Dict, List

from controlplane.config import settings
from controlplane.retrievers.base import RetrievedChunk
from controlplane.retrievers.registry import get_lc_minilm


def _tok(text: str) -> List[str]:
    return [t for t in re.sub(r"[^\w\s-]", " ", (text or "").lower()).split() if len(t) > 1]


class HRPolicyKB:
    kb_id = "hr_policy"

    def __init__(self) -> None:
        self._faiss = None
        self._parents: Dict[str, dict] = {}
        self._bm25 = None
        self._parent_ids: List[str] = []

    def load(self) -> None:
        if self._faiss is not None:
            return
        from langchain_community.vectorstores import FAISS

        p = settings.paths
        self._faiss = FAISS.load_local(
            str(p["hr_faiss"]), get_lc_minilm(normalize=False), allow_dangerous_deserialization=True
        )
        with open(p["hr_parents"], "r", encoding="utf-8") as f:
            self._parents = json.load(f)

        if p["hr_bm25"].exists():
            with open(p["hr_bm25"], "rb") as f:
                data = pickle.load(f)
            self._bm25 = data["bm25"]
            self._parent_ids = data["parent_ids"]
        else:
            self._build_bm25_in_memory()

    def _build_bm25_in_memory(self) -> None:
        from rank_bm25 import BM25Okapi

        self._parent_ids = list(self._parents.keys())
        corpus = [
            _tok(f"{self._parents[pid].get('section_title', '')} {self._parents[pid].get('content', '')}")
            for pid in self._parent_ids
        ]
        self._bm25 = BM25Okapi(corpus)

    def _parent_chunk(self, pid: str, rtype: str, score: float = 0.0) -> RetrievedChunk:
        par = self._parents.get(pid, {})
        return RetrievedChunk(
            chunk_id=pid,
            text=par.get("content", ""),
            source=par.get("source", "hr_policy"),
            title=par.get("section_title", "HR Policy"),
            score=score,
            metadata={"section_title": par.get("section_title", ""), "parent_id": pid},
            retrieval_types=[rtype],
        )

    def vector_search(self, query: str, k: int) -> List[RetrievedChunk]:
        self.load()
        children = self._faiss.similarity_search(query, k=max(k * 2, k + 2))
        seen: List[str] = []
        out: List[RetrievedChunk] = []
        for child in children:
            pid = (child.metadata or {}).get("parent_id")
            if pid and pid not in seen and pid in self._parents:
                seen.append(pid)
                out.append(self._parent_chunk(pid, "vector"))
            if len(out) >= k:
                break
        return out

    def bm25_search(self, query: str, k: int) -> List[RetrievedChunk]:
        self.load()
        scores = self._bm25.get_scores(_tok(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out: List[RetrievedChunk] = []
        for i in ranked[:k]:
            if scores[i] <= 0:
                continue
            out.append(self._parent_chunk(self._parent_ids[i], "bm25", float(scores[i])))
        return out
