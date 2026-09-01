"""
Human-in-the-loop node.

`hitl_interrupt` calls LangGraph's `interrupt()` with the question + context; the
graph pauses and the Streamlit UI shows exactly what is needed. On resume the
user's text is **merged into the query** and every downstream field is reset, so
the graph re-enters at `guardrails` and runs the *identical* full pipeline on the
enriched query (guardrails -> cache -> router -> retrieval -> answer -> both
evaluation branches -> verdict). `hitl_count` is bumped to 1 (enforced once).
"""

from __future__ import annotations

import os
from typing import Any, Dict

from controlplane.observability import traceable_node
from controlplane.state import Stage

_DEBUG = os.getenv("CP_DEBUG", "").lower() in {"1", "true", "yes"}

# every field a fresh pass would (re)compute - cleared so nothing stale leaks in
_RESET_FIELDS = {
    "cache_hit": False, "cache_similarity": None, "cached_answer": None, "cached_meta": None,
    "selected_kb": None, "router_reason": None, "router_confidence": None,
    "router_semantic_scores": {},
    "vector_chunks": [], "bm25_chunks": [], "rrf_chunks": [], "retrieval_meta": {},
    "answer": "", "original_answer": None, "edit_reason": None,
    "model_used": None, "model_category": None, "model_tier": None,
    "token_stats": {}, "cost_usd": 0.0,
    "ragas_scores": {}, "ragas_verdict": None, "ragas_unsupported": [],
    "xgboost_prob": None, "xgboost_risk": None, "xgboost_features": {},
    "entity_drift": {}, "perf_verdict": None, "perf_reasoning": None,
    "perf_suggestion": None, "perf_score": None, "detector_votes": {},
    "resp_vector_chunks": [], "resp_bm25_chunks": [], "resp_graph_chunks": [],
    "resp_rrf_chunks": [], "resp_retrieval_meta": {},
    "toxicity": {}, "toxicity_max": None, "resp_status": None, "resp_reasoning": None,
    "violated_rules": [], "resp_report": None, "evidence_chunks": [],
    "final_decision": None, "final_verdict": None, "final_answer": "",
    "final_verdict_badges": [], "hitl_needed": False, "hitl_question": None,
    "blocked": False, "block_reason": None, "block_category": None,
}


@traceable_node("hitl_interrupt")
def hitl_interrupt_node(state: Dict[str, Any]) -> Dict[str, Any]:
    from langgraph.types import interrupt

    base_query = state.get("pre_hitl_query") or state.get("original_query", "")
    payload = {
        "question": state.get("hitl_question")
        or "More information is required to answer this reliably.",
        "original_query": base_query,
        "selected_kb": state.get("selected_kb", ""),
        "draft_answer": state.get("answer", ""),
        "perf_reasoning": state.get("perf_reasoning", ""),
        "perf_verdict": state.get("perf_verdict", ""),
        "resp_status": state.get("resp_status", ""),
        "ragas_scores": state.get("ragas_scores", {}),
    }
    user_reply = interrupt(payload)  # <-- execution pauses here until resume

    reply = (user_reply or "").strip() if isinstance(user_reply, str) else str(user_reply or "")
    enriched = f"{base_query}\n\n[Additional information from the user: {reply}]" if reply else base_query

    if _DEBUG:
        print(f"[cp:hitl] reply={reply[:100]!r}  -> enriched query re-runs the full pipeline", flush=True)

    out: Dict[str, Any] = dict(_RESET_FIELDS)
    out.update({
        "stage": Stage.HITL,
        "stages_visited": [Stage.HITL],
        "hitl_count": int(state.get("hitl_count", 0)) + 1,
        "hitl_response": reply,
        "pre_hitl_query": base_query,        # keep the un-enriched query for the UI / audit
        # the enriched text becomes the query every node downstream sees
        "original_query": enriched,
        "guarded_query": enriched,
        "updated_query": enriched,
        "retry_count": 0,                    # the enriched query gets its own retry budget
    })
    return out
