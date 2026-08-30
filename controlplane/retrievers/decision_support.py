"""
Decision Support KB adapter (corporate meeting transcripts).

Reuses the Data/ artefacts built by the original project:
  * faiss_index.bin        - native FAISS IndexFlatIP over normalised MiniLM child vectors
  * children_store.json    - list[ {child_id, parent_id, meeting_id, speaker, text} ]  (index-aligned to FAISS + BM25)
  * parents_store.json     - dict{ parent_id -> {meeting_id, speaker, text, metadata} }
  * bm25_index.pkl         - {"bm25": BM25Okapi}  scored over the same child list

Both branches retrieve child hits and resolve to unique PARENT chunks so the
answer generator gets full dialogue context.
"""

from __future__ import annotations

import json
import pickle
from typing import Dict, List

from controlplane.config import settings
from controlplane.retrievers.base import RetrievedChunk
from controlplane.retrievers.registry import get_minilm


class DecisionSupportKB:
    kb_id = "decision_support"

    def __init__(self) -> None:
        self._index = None
        self._children: List[dict] = []
        self._parents: Dict[str, dict] = {}
        self._bm25 = None
        self._embed = None

    def load(self) -> None:
        if self._index is not None:
            return
        import faiss

        d = settings.paths["ds_dir"]
        self._index = faiss.read_index(str(d / "faiss_index.bin"))
        with open(d / "children_store.json", "r", encoding="utf-8") as f:
            self._children = json.load(f)
        with open(d / "parents_store.json", "r", encoding="utf-8") as f:
            self._parents = json.load(f)
        with open(d / "bm25_index.pkl", "rb") as f:
            self._bm25 = pickle.load(f)["bm25"]
        self._embed = get_minilm()

    def _resolve_parents(self, child_indices: List[int], scores: List[float], rtype: str, k: int) -> List[RetrievedChunk]:
        out: List[RetrievedChunk] = []
        seen: set[str] = set()
        for idx, sc in zip(child_indices, scores):
            if idx < 0 or idx >= len(self._children):
                continue
            child = self._children[idx]
            pid = child.get("parent_id")
            if not pid or pid in seen:
                continue
            seen.add(pid)
            par = self._parents.get(pid, {})
            meta = par.get("metadata", {}) if isinstance(par.get("metadata"), dict) else {}
            out.append(
                RetrievedChunk(
                    chunk_id=pid,
                    text=par.get("text", child.get("text", "")),
                    source=f"meeting {par.get('meeting_id', child.get('meeting_id', '?'))}",
                    title=f"{par.get('speaker', child.get('speaker', 'Speaker'))} - meeting {par.get('meeting_id', '?')}",
                    score=float(sc),
                    metadata={"meeting_id": par.get("meeting_id"), "matched_child": child.get("text", "")[:200], **meta},
                    retrieval_types=[rtype],
                )
            )
            if len(out) >= k:
                break
        return out

    def vector_search(self, query: str, k: int) -> List[RetrievedChunk]:
        self.load()
        import faiss
        import numpy as np

        q = self._embed.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q)
        pool = min(len(self._children), max(k * 8, 40))
        dist, idx = self._index.search(q, pool)
        return self._resolve_parents(idx[0].tolist(), dist[0].tolist(), "vector", k)

    def bm25_search(self, query: str, k: int) -> List[RetrievedChunk]:
        self.load()
        import numpy as np

        scores = np.asarray(self._bm25.get_scores(query.lower().split()))
        pool = min(len(scores), max(k * 8, 40))
        top = np.argpartition(scores, -pool)[-pool:]
        top = top[np.argsort(scores[top])[::-1]]
        return self._resolve_parents(top.tolist(), scores[top].tolist(), "bm25", k)
