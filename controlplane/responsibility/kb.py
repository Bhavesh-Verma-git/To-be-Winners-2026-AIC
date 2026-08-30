"""
Responsibility knowledge base - 3-way hybrid retrieval over the compliance corpus
(EU AI Act, NIST AI RMF, UN/UNESCO hate-speech, EU DSA, EEOC, CoE, OHCHR).

Branches (run in parallel by the node):
  * vector : ChromaDB collection rebuilt locally with MiniLM
             (`scripts/build_responsibility_index.py`; the shipped chroma_db was
             hash-embedded and unusable).  Falls back to an in-memory MiniLM
             matrix over chunk_store.json if the rebuilt collection is missing.
  * bm25   : the shipped `data/bm25_index.pkl` ({bm25, chunk_ids}) - embedder independent.
  * graph  : Neo4j (env-configured) if reachable, else token-scoring over the
             shipped `data/graph_triples.json` (same logic as the original
             `graph_store.query_graph_for_chunks`).

Then Reciprocal Rank Fusion -> top-k evidence chunks. NO LLM in this module.
"""

from __future__ import annotations

import json
import pickle
import re
import threading
from typing import Any, Dict, List, Optional

from controlplane.config import settings
from controlplane.retrievers.base import RetrievedChunk, rrf_fuse
from controlplane.retrievers.registry import get_minilm

_COLLECTION = "compliance_local_minilm"


def _chunk_to_retrieved(cid: str, rec: dict, rtype: str, score: float = 0.0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid,
        text=rec.get("text", ""),
        source=rec.get("source_file", rec.get("doc_title", "")),
        title=rec.get("law_or_article") or rec.get("heading_hierarchy") or rec.get("doc_title", ""),
        score=score,
        metadata={
            "law_or_article": rec.get("law_or_article", ""),
            "doc_title": rec.get("doc_title", ""),
            "heading_hierarchy": rec.get("heading_hierarchy", ""),
            "page_numbers": rec.get("page_numbers", []),
        },
        retrieval_types=[rtype],
    )


class ResponsibilityKB:
    def __init__(self) -> None:
        self._store: Dict[str, dict] = {}
        self._bm25 = None
        self._bm25_ids: List[str] = []
        self._chroma = None
        self._matrix = None
        self._matrix_ids: List[str] = []
        self._triples: List[dict] = []
        self._neo4j = None
        self._lock = threading.Lock()
        self._loaded = False

    # ---- load -------------------------------------------------------------
    def load(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            p = settings.paths
            with open(p["resp_chunks"], "r", encoding="utf-8") as f:
                self._store = json.load(f)
            if p["resp_bm25"].exists():
                with open(p["resp_bm25"], "rb") as f:
                    data = pickle.load(f)
                self._bm25, self._bm25_ids = data["bm25"], data["chunk_ids"]
            if p["resp_triples"].exists():
                with open(p["resp_triples"], "r", encoding="utf-8") as f:
                    self._triples = json.load(f)
            self._load_vector()
            self._connect_neo4j()
            self._loaded = True

    def _load_vector(self) -> None:
        p = settings.paths
        # 1) rebuilt Chroma collection (scripts/build_responsibility_index.py)
        try:
            import chromadb

            if p["resp_chroma_local"].exists():
                client = chromadb.PersistentClient(path=str(p["resp_chroma_local"]))
                self._chroma = client.get_collection(_COLLECTION)
                if self._chroma.count() > 0:
                    return
                self._chroma = None
        except Exception:
            self._chroma = None

        # 2) prebuilt MiniLM matrix (.npz) - no chromadb needed
        try:
            import numpy as np

            if p["resp_matrix"].exists():
                data = np.load(p["resp_matrix"], allow_pickle=True)
                self._matrix = data["matrix"].astype("float32")
                self._matrix_ids = list(data["ids"])
                return
        except Exception:
            self._matrix = None

        # 3) live embed at load time (a few seconds, once)
        try:
            import numpy as np

            model = get_minilm()
            if model is None:
                return
            self._matrix_ids = list(self._store.keys())
            texts = [self._store[c].get("text", "") for c in self._matrix_ids]
            self._matrix = np.asarray(
                model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False),
                dtype="float32",
            )
        except Exception:
            self._matrix = None

    def _connect_neo4j(self) -> None:
        try:
            from controlplane.responsibility.neo4j_util import get_driver

            self._neo4j = get_driver(verify=True)
        except Exception:
            self._neo4j = None

    # ---- branches -------------------------------------------------------------
    def vector_search(self, query: str, k: int) -> List[RetrievedChunk]:
        self.load()
        model = get_minilm()
        if model is None:
            return []
        q = model.encode([query], normalize_embeddings=True)[0]

        if self._chroma is not None:
            try:
                res = self._chroma.query(query_embeddings=[q.tolist()], n_results=k)
                ids = (res.get("ids") or [[]])[0]
                dists = (res.get("distances") or [[None] * len(ids)])[0]
                out = []
                for cid, dist in zip(ids, dists):
                    rec = self._store.get(cid, {})
                    score = 1.0 - float(dist) if dist is not None else 0.0
                    out.append(_chunk_to_retrieved(cid, rec, "vector", score))
                return out
            except Exception:
                pass

        if self._matrix is not None:
            import numpy as np

            sims = self._matrix @ np.asarray(q, dtype="float32")
            top = np.argsort(sims)[::-1][:k]
            return [
                _chunk_to_retrieved(self._matrix_ids[i], self._store.get(self._matrix_ids[i], {}), "vector", float(sims[i]))
                for i in top
            ]
        return []

    def bm25_search(self, query: str, k: int) -> List[RetrievedChunk]:
        self.load()
        if self._bm25 is None:
            return []
        toks = [t for t in re.sub(r"[^\w\s-]", " ", query.lower()).split() if len(t) > 1]
        scores = self._bm25.get_scores(toks)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        out = []
        for i in ranked:
            if scores[i] <= 0:
                continue
            cid = self._bm25_ids[i]
            out.append(_chunk_to_retrieved(cid, self._store.get(cid, {}), "bm25", float(scores[i])))
        return out

    def graph_search(self, query: str, k: int) -> List[RetrievedChunk]:
        self.load()
        matched: List[str] = []

        if self._neo4j is not None:
            try:
                cypher = (
                    "MATCH (n:Entity) WHERE toLower(n.name) CONTAINS $t "
                    "OR toLower(coalesce(n.law_or_article,'')) CONTAINS $t "
                    "RETURN DISTINCT n.chunk_id AS cid LIMIT $lim"
                )
                with self._neo4j.session(database=settings.neo4j_database) as s:
                    for tok in {t.lower() for t in re.findall(r"\w+", query) if len(t) > 3}:
                        for rec in s.run(cypher, t=tok, lim=k):
                            cid = rec["cid"]
                            if cid and cid not in matched:
                                matched.append(cid)
                        if len(matched) >= k:
                            break
            except Exception:
                pass

        if len(matched) < k and self._triples:
            toks = [t.lower() for t in re.findall(r"\w+", query) if len(t) > 2]
            scored: Dict[str, int] = {}
            for tr in self._triples:
                blob = f"{tr.get('source', '')} {tr.get('target', '')} {tr.get('law_or_article', '')} {tr.get('heading_path', '')}".lower()
                cid = tr.get("chunk_id")
                if not cid:
                    continue
                hit = sum(
                    (4 if t in (tr.get("law_or_article", "") or "").lower() else 0)
                    + (3 if t in blob else 0)
                    for t in toks
                )
                if hit:
                    scored[cid] = scored.get(cid, 0) + hit
            for cid in sorted(scored, key=scored.get, reverse=True):
                if cid not in matched:
                    matched.append(cid)
                if len(matched) >= k:
                    break

        return [
            _chunk_to_retrieved(cid, self._store.get(cid, {}), "graph")
            for cid in matched[:k]
            if cid in self._store
        ]

    # ---- orchestration -------------------------------------------------------------
    async def retrieve(self, query: str) -> Dict[str, Any]:
        import asyncio
        import time

        kpb = settings.responsibility_top_k_per_branch
        t0 = time.perf_counter()
        vec, bm, gr = await asyncio.gather(
            asyncio.to_thread(self.vector_search, query, kpb),
            asyncio.to_thread(self.bm25_search, query, kpb),
            asyncio.to_thread(self.graph_search, query, kpb),
        )
        # 3-way RRF (pairwise fuse then fuse again is equivalent to full RRF for equal weights)
        fused_pair = rrf_fuse(vec, bm, top_k=max(len(vec) + len(bm), settings.responsibility_rrf_top_k))
        fused = rrf_fuse(fused_pair, gr, top_k=settings.responsibility_rrf_top_k)
        return {
            "vector_chunks": [c.to_dict() for c in vec],
            "bm25_chunks": [c.to_dict() for c in bm],
            "graph_chunks": [c.to_dict() for c in gr],
            "rrf_chunks": [c.to_dict() for c in fused],
            "meta": {
                "neo4j": self._neo4j is not None,
                "vector_backend": "chroma" if self._chroma is not None else ("matrix" if self._matrix is not None else "none"),
                "total_ms": round((time.perf_counter() - t0) * 1000, 1),
            },
        }


_kb: Optional[ResponsibilityKB] = None


def get_responsibility_kb() -> ResponsibilityKB:
    global _kb
    if _kb is None:
        _kb = ResponsibilityKB()
        _kb.load()
    return _kb
