"""Customer Support KB adapter - reuses the existing FAISS (MiniLM) + pickled BM25Retriever."""

from __future__ import annotations

import pickle
from typing import List

from controlplane.config import settings
from controlplane.retrievers.base import RetrievedChunk
from controlplane.retrievers.registry import get_lc_minilm


class CustomerSupportKB:
    kb_id = "customer_support"

    def __init__(self) -> None:
        self._faiss = None
        self._bm25 = None

    def load(self) -> None:
        if self._faiss is not None:
            return
        from langchain_community.vectorstores import FAISS

        p = settings.paths
        self._faiss = FAISS.load_local(
            str(p["cs_faiss"]), get_lc_minilm(normalize=False), allow_dangerous_deserialization=True
        )
        with open(p["cs_bm25"], "rb") as f:
            self._bm25 = pickle.load(f)

    def _to_chunk(self, doc, rtype: str) -> RetrievedChunk:
        md = dict(doc.metadata or {})
        answer = md.get("response") or doc.page_content
        return RetrievedChunk(
            chunk_id=str(md.get("id") or md.get("intent", "") + "|" + doc.page_content[:40]),
            text=f"Customer: {doc.page_content}\nResolution: {answer}",
            source=md.get("source", "customer_support"),
            title=f"{md.get('category', 'SUPPORT')} / {md.get('intent', 'general')}",
            metadata=md,
            retrieval_types=[rtype],
        )

    def vector_search(self, query: str, k: int) -> List[RetrievedChunk]:
        self.load()
        return [self._to_chunk(d, "vector") for d in self._faiss.similarity_search(query, k=k)]

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
        return [self._to_chunk(d, "bm25") for d in docs[:k]]
