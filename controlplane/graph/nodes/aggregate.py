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

# don't start a retry/HITL loop if we've already spent this long - protects the 10s ceiling
_RETRY_DEADLINE_S = float(os.getenv("CP_RETRY_DEADLINE_S", "5.0"))


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

    out: Dict[str, Any] = {
        "stage": Stage.AGGREGATE,
        "stages_visited": [Stage.AGGREGATE],
        "cost_usd": round(cost, 6),
    }

    if resp == "unsafe":
        out["_next"] = "harmful"
        out["final_decision"] = "harmful"
        return out

    if perf == "hallucinated" and retry_count < settings.max_hallucination_retries and budget_left:
        suggestion = state.get("perf_suggestion") or state.get("updated_query") or state.get("guarded_query")
        out["_next"] = "retry"
        out["retry_count"] = retry_count + 1
        out["updated_query"] = suggestion
        out["stage"] = Stage.HALLUCINATION_RETRY
        out["stages_visited"] = [Stage.AGGREGATE, Stage.HALLUCINATION_RETRY]
        return out

    if (hitl_needed or perf == "need_human" or resp == "uncertain") and hitl_count < settings.max_hitl_rounds:
        # HITL pauses (human time doesn't count) so no budget check here
        out["_next"] = "hitl"
        out["hitl_needed"] = True
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
