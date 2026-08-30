"""Azure App Service docs KB adapter - FAISS (bge-small) + BM25Okapi over chunks.jsonl."""

from __future__ import annotations

import json
import pickle
from typing import Dict, List

from controlplane.config import settings
from controlplane.retrievers.base import RetrievedChunk
from controlplane.retrievers.registry import get_lc_bge


class InternalKnowledgeKB:
    kb_id = "internal_knowledge"

    def __init__(self) -> None:
        self._faiss = None
        self._bm25 = None
        self._chunk_ids: List[str] = []
        self._store: Dict[str, dict] = {}

    def load(self) -> None:
        if self._faiss is not None:
            return
        from langchain_community.vectorstores import FAISS

        p = settings.paths
        self._faiss = FAISS.load_local(
            str(p["ik_faiss"]), get_lc_bge(), allow_dangerous_deserialization=True
        )
        with open(p["ik_bm25"], "rb") as f:
            data = pickle.load(f)
        self._bm25 = data["bm25"]
        self._chunk_ids = data["chunk_ids"]
        with open(p["ik_chunks"], "r", encoding="utf-8") as f:
            for line in f:
                c = json.loads(line)
                self._store[c["chunk_id"]] = c

    def _chunk(self, c: dict, rtype: str, score: float = 0.0) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=c["chunk_id"],
            text=c.get("text", ""),
            source=c.get("source_url", c.get("source", "")),
            title=f"{c.get('title', '')} > {c.get('section', '')}".strip(" >"),
            score=score,
            metadata={
                "source_url": c.get("source_url", ""),
                "section": c.get("section", ""),
                "has_code": c.get("has_code", False),
            },
            retrieval_types=[rtype],
        )

    def vector_search(self, query: str, k: int) -> List[RetrievedChunk]:
        self.load()
        out: List[RetrievedChunk] = []
        for d in self._faiss.similarity_search(query, k=k):
            cid = (d.metadata or {}).get("chunk_id")
            c = self._store.get(cid)
            if c:
                out.append(self._chunk(c, "vector"))
            else:
                out.append(
                    RetrievedChunk(chunk_id=str(cid), text=d.page_content, retrieval_types=["vector"])
                )
        return out

    def bm25_search(self, query: str, k: int) -> List[RetrievedChunk]:
        self.load()
        scores = self._bm25.get_scores(query.lower().split())
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        out: List[RetrievedChunk] = []
        for i in ranked:
            if scores[i] <= 0:
                continue
            c = self._store.get(self._chunk_ids[i])
            if c:
                out.append(self._chunk(c, "bm25", float(scores[i])))
        return out
