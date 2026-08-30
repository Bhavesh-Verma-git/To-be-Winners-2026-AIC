"""
Shared retrieval primitives.

* `RetrievedChunk` - one normalised hit (every KB adapter emits these).
* `HybridKB` - the interface every adapter implements: `vector_search` + `bm25_search`.
* `rrf_fuse` - Reciprocal Rank Fusion (k=60) over the two ranked lists.
* `hybrid_retrieve` - runs the two searches IN PARALLEL (threads) then fuses.

Lifted / unified from the RRF implementations already in the repo
(`rag_agents/internal_knowledge/rag_agent.py`, `Responsiblity Agent/src/retrieval/rrf.py`).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol, Sequence

from controlplane.config import settings


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    source: str = ""
    title: str = ""
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    retrieval_types: List[str] = field(default_factory=list)   # ["vector"] / ["bm25"] / both
    ranks: Dict[str, int] = field(default_factory=dict)         # {"vector": 1, "bm25": 3}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source": self.source,
            "title": self.title,
            "score": round(self.score, 5),
            "metadata": self.metadata,
            "retrieval_types": self.retrieval_types,
            "ranks": self.ranks,
        }


class HybridKB(Protocol):
    kb_id: str

    def vector_search(self, query: str, k: int) -> List[RetrievedChunk]: ...
    def bm25_search(self, query: str, k: int) -> List[RetrievedChunk]: ...


def rrf_fuse(
    vector_hits: Sequence[RetrievedChunk],
    bm25_hits: Sequence[RetrievedChunk],
    *,
    k_constant: int | None = None,
    top_k: int | None = None,
) -> List[RetrievedChunk]:
    """Reciprocal Rank Fusion. score(d) = sum_r 1 / (k + rank_r(d)), 1-based ranks."""
    k_constant = k_constant or settings.rrf_k_constant
    top_k = top_k or settings.rrf_top_k

    scores: Dict[str, float] = {}
    merged: Dict[str, RetrievedChunk] = {}

    for label, hits in (("vector", vector_hits), ("bm25", bm25_hits)):
        for rank_idx, hit in enumerate(hits):
            rank = rank_idx + 1
            cid = hit.chunk_id or f"{label}:{rank}:{hit.text[:24]}"
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k_constant + rank)
            if cid not in merged:
                merged[cid] = RetrievedChunk(
                    chunk_id=cid,
                    text=hit.text,
                    source=hit.source,
                    title=hit.title,
                    metadata=dict(hit.metadata),
                )
            merged[cid].retrieval_types.append(label)
            merged[cid].ranks[label] = rank

    ordered = sorted(scores, key=lambda c: scores[c], reverse=True)[:top_k]
    out: List[RetrievedChunk] = []
    for cid in ordered:
        chunk = merged[cid]
        chunk.score = round(scores[cid], 6)
        out.append(chunk)
    return out


async def hybrid_retrieve(kb: HybridKB, query: str) -> Dict[str, Any]:
    """Vector top-k  ||  BM25 top-k  (parallel)  ->  RRF  ->  top-k fused chunks."""
    t0 = time.perf_counter()

    async def _vec():
        s = time.perf_counter()
        hits = await asyncio.to_thread(kb.vector_search, query, settings.vector_top_k)
        return hits, (time.perf_counter() - s) * 1000

    async def _bm25():
        s = time.perf_counter()
        hits = await asyncio.to_thread(kb.bm25_search, query, settings.bm25_top_k)
        return hits, (time.perf_counter() - s) * 1000

    (vec_hits, vec_ms), (bm25_hits, bm25_ms) = await asyncio.gather(_vec(), _bm25())
    fused = rrf_fuse(vec_hits, bm25_hits)
    total_ms = (time.perf_counter() - t0) * 1000

    return {
        "vector_chunks": [h.to_dict() for h in vec_hits],
        "bm25_chunks": [h.to_dict() for h in bm25_hits],
        "rrf_chunks": [h.to_dict() for h in fused],
        "meta": {
            "kb": getattr(kb, "kb_id", "unknown"),
            "vector_ms": round(vec_ms, 1),
            "bm25_ms": round(bm25_ms, 1),
            "total_ms": round(total_ms, 1),
            "vector_n": len(vec_hits),
            "bm25_n": len(bm25_hits),
            "fused_n": len(fused),
        },
    }


def hybrid_retrieve_sync(kb: HybridKB, query: str) -> Dict[str, Any]:
    return asyncio.run(hybrid_retrieve(kb, query))


def format_context(chunks: Sequence[Dict[str, Any] | RetrievedChunk], max_chars: int = 6000) -> str:
    """Render fused chunks into a grounding block for the answer generator."""
    parts: List[str] = []
    budget = max_chars
    for i, c in enumerate(chunks, 1):
        d = c.to_dict() if isinstance(c, RetrievedChunk) else c
        head = d.get("title") or d.get("source") or f"chunk {i}"
        body = (d.get("text") or "").strip()
        block = f"[{i}] {head}\n{body}"
        if len(block) > budget:
            block = block[: max(0, budget)] + " ..."
        parts.append(block)
        budget -= len(block)
        if budget <= 0:
            break
    return "\n\n---\n\n".join(parts)
