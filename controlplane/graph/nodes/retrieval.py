"""Retrieval node - vector top-5  ||  BM25 top-5  ->  RRF  ->  top-5 (parallel)."""

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict

from controlplane.observability import traceable_node
from controlplane.retrievers import get_kb, hybrid_retrieve
from controlplane.state import Stage

_DEBUG = os.getenv("CP_DEBUG", "").lower() in {"1", "true", "yes"}


@traceable_node("retrieval")
def retrieval_node(state: Dict[str, Any]) -> Dict[str, Any]:
    kb_id = state.get("selected_kb") or "customer_support"
    query = state.get("updated_query") or state.get("guarded_query") or state.get("original_query", "")

    if kb_id == "none":
        return {
            "stage": Stage.RETRIEVAL,
            "stages_visited": [Stage.RETRIEVAL],
            "vector_chunks": [],
            "bm25_chunks": [],
            "rrf_chunks": [],
            "retrieval_meta": {"kb": "none", "note": "no knowledge base matched"},
        }

    kb = get_kb(kb_id)
    result = asyncio.run(hybrid_retrieve(kb, query))
    if _DEBUG:
        top = (result["rrf_chunks"][0]["text"][:90] if result["rrf_chunks"] else "-")
        print(f"[cp:retrieval] kb={kb_id} q={query[:70]!r} "
              f"vector_n={result['meta']['vector_n']} bm25_n={result['meta']['bm25_n']} "
              f"fused_n={result['meta']['fused_n']} top={top!r}", flush=True)
    return {
        "stage": Stage.RETRIEVAL,
        "stages_visited": [Stage.RETRIEVAL],
        "vector_chunks": result["vector_chunks"],
        "bm25_chunks": result["bm25_chunks"],
        "rrf_chunks": result["rrf_chunks"],
        "retrieval_meta": result["meta"],
    }
