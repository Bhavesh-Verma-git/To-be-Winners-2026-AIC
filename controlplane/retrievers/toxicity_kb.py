"""
Toxicity / content-safety KB adapter.

Wraps the Toxic_RAG index cache: FAISS (MiniLM, normalised) + pickled
BM25Retriever over the annotated toxic-statement corpus. Each hit carries the
safety annotations (target group, framing, stereotyping, factual, lewd).
"""

from __future__ import annotations

import pickle
from typing import List

from controlplane.config import settings
from controlplane.retrievers.base import RetrievedChunk
from controlplane.retrievers.registry import get_lc_minilm

_ANNOT_KEYS = ["target group", "factual", "in-group effect", "framing", "lewd", "predicted group", "stereotyping"]


class ToxicityKB:
    kb_id = "toxicity_kb"

    def __init__(self) -> None:
        self._faiss = None
        self._bm25 = None

    def load(self) -> None:
        if self._faiss is not None:
            return
        from langchain_community.vectorstores import FAISS

        p = settings.paths
        self._faiss = FAISS.load_local(
            str(p["tox_faiss"]), get_lc_minilm(normalize=True), allow_dangerous_deserialization=True
        )
        with open(p["tox_bm25"], "rb") as f:
            self._bm25 = pickle.load(f)

    def _chunk(self, doc, rtype: str) -> RetrievedChunk:
        md = dict(doc.metadata or {})
        annot = ", ".join(f"{key}={md[key]}" for key in _ANNOT_KEYS if md.get(key))
        return RetrievedChunk(
            chunk_id=str(md.get("row_id", doc.page_content[:40])),
            text=f'"{doc.page_content}"' + (f"  [annotations: {annot}]" if annot else ""),
            source="toxicity_kb",
            title=f"target={md.get('target group', 'n/a')}",
            metadata=md,
            retrieval_types=[rtype],
        )

    def vector_search(self, query: str, k: int) -> List[RetrievedChunk]:
        self.load()
        return [self._chunk(d, "vector") for d in self._faiss.similarity_search(query, k=k)]

    def bm25_search(self, query: str, k: int) -> List[RetrievedChunk]:
        self.load()
        try:
            self._bm25.k = k
        except Exception:
            pass
        try:
            docs = self._bm25.invoke(query)
        except Exception:
            docs = self._bm25.get_relevant_documents(query)
        return [self._chunk(d, "bm25") for d in docs[:k]]
