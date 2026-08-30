"""Semantic cache node. Runs after guardrails, before the RAG router."""

from __future__ import annotations

from typing import Any, Dict

from controlplane.cache import get_cache
from controlplane.observability import traceable_node
from controlplane.state import Stage


@traceable_node("semantic_cache")
def semantic_cache_node(state: Dict[str, Any]) -> Dict[str, Any]:
    query = state.get("guarded_query") or state.get("original_query", "")
    hit = get_cache().lookup(query)

    if hit is None:
        return {"stage": Stage.CACHE, "stages_visited": [Stage.CACHE], "cache_hit": False}

    return {
        "stage": Stage.CACHE,
        "stages_visited": [Stage.CACHE],
        "cache_hit": True,
        "cache_similarity": round(hit.similarity, 4),
        "cached_answer": hit.answer,
        "cached_meta": hit.meta,
        "selected_kb": hit.meta.get("selected_kb"),
        "final_decision": "cache",
    }


def route_after_cache(state: Dict[str, Any]) -> str:
    return "hit" if state.get("cache_hit") else "miss"
