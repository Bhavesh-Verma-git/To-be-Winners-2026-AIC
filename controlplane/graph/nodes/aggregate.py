"""
Aggregate / decision node - the join where both parallel branches meet.

Decision order (safety-biased):
  1. responsibility UNSAFE           -> harmful   (short-circuits retry/HITL)
  2. performance HALLUCINATED, retry<1 -> retry    (updated_query <- suggestion, retry_count++)
  3. need-human OR responsibility uncertain, hitl<1 -> hitl
  4. otherwise                       -> safe

Also rolls up total cost from `llm_calls` (Groq already 0) so no parallel node
ever writes `cost_usd` concurrently.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict

from controlplane.config import settings
from controlplane.observability import traceable_node
from controlplane.state import Stage

_DEBUG = os.getenv("CP_DEBUG", "").lower() in {"1", "true", "yes"}
# don't start a self-reflection retry if we've already spent this long - the lean
# retry pass (skips responsibility re-retrieval + RAGAS, caps the answer) adds
# ~2-2.5s, so this keeps the total under the 10s ceiling.
_RETRY_DEADLINE_S = float(os.getenv("CP_RETRY_DEADLINE_S", "7.0"))

# markers that mean "the first pass produced NO usable answer" - when perf flags
# one of these the retry runs even if the latency budget is spent: a correct
# answer a few seconds late always beats a fast "no information" reply.
_NONANSWER_MARKERS = (
    "does not contain enough information", "not contain enough information",
    "not enough information", "no relevant information", "no information",
    "cannot answer", "unable to answer",
)


@traceable_node("aggregate")
def aggregate_node(state: Dict[str, Any]) -> Dict[str, Any]:
    perf = state.get("perf_verdict") or "pass"
    resp = state.get("resp_status") or "safe"
    retry_count = int(state.get("retry_count", 0))
    hitl_count = int(state.get("hitl_count", 0))
    hitl_needed = bool(state.get("hitl_needed"))

    cost = sum(float(c.get("cost_usd", 0.0) or 0.0) for c in (state.get("llm_calls", []) or []))
    elapsed = time.time() - float(state.get("started_at", time.time()))
    budget_left = elapsed < _RETRY_DEADLINE_S  # room for another retrieval+answer+eval pass?

    # a first pass that returned no usable answer always earns its one retry, even
    # over budget - otherwise a slow host just shows the "no information" draft.
    _ans_low = (state.get("answer", "") or "").lower()
    is_nonanswer = any(m in _ans_low for m in _NONANSWER_MARKERS)
    retry_ok = budget_left or is_nonanswer

    if _DEBUG:
        print(f"[cp:aggregate] perf={perf} resp={resp} retry={retry_count} hitl={hitl_count} "
              f"elapsed={elapsed:.1f}s budget_left={budget_left} nonanswer={is_nonanswer}", flush=True)

    out: Dict[str, Any] = {
        "stage": Stage.AGGREGATE,
        "stages_visited": [Stage.AGGREGATE],
        "cost_usd": round(cost, 6),
    }

    if resp == "unsafe":
        out["_next"] = "harmful"
        out["final_decision"] = "harmful"
        out["final_verdict"] = "BLOCK"
        return out

    if perf == "hallucinated" and retry_count < settings.max_hallucination_retries and retry_ok:
        suggestion = state.get("perf_suggestion") or state.get("updated_query") or state.get("guarded_query")
        out["_next"] = "retry"
        out["retry_count"] = retry_count + 1
        # preserve the pre-EDIT draft + the reason so the UI can show the before/after
        out["original_answer"] = state.get("answer", "")
        out["edit_reason"] = state.get("perf_reasoning", "")
        out["updated_query"] = suggestion
        out["final_verdict"] = "EDIT — self-reflection"
        out["stage"] = Stage.HALLUCINATION_RETRY
        out["stages_visited"] = [Stage.AGGREGATE, Stage.HALLUCINATION_RETRY]
        if _DEBUG:
            print(f"[cp:aggregate] EDIT -> re-retrieve with: {str(suggestion)[:120]!r}", flush=True)
        return out

    if (hitl_needed or perf == "need_human" or resp == "uncertain") and hitl_count < settings.max_hitl_rounds:
        # HITL pauses (human time doesn't count) so no budget check here
        out["_next"] = "hitl"
        out["hitl_needed"] = True
        out["final_verdict"] = "HUMAN-IN-THE-LOOP"
        if not state.get("hitl_question"):
            out["hitl_question"] = (
                "I need more detail to answer this reliably. Please specify the exact "
                "product / policy section / time period / identifiers involved."
            )
        return out

    out["_next"] = "safe"
    out["final_decision"] = "allow"
    return out


def route_after_aggregate(state: Dict[str, Any]) -> str:
    return state.get("_next", "safe")
