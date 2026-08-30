"""
Terminal nodes - format the response for each outcome, write the cache on the
safe path, and stamp total latency.

  finalize_block   - guardrail rejection
  finalize_cache   - semantic-cache hit
  finalize_harmful - responsibility flagged the answer as unsafe
  finalize_safe    - normal answer (cache write happens here)
"""

from __future__ import annotations

import time
from typing import Any, Dict

from controlplane.cache import get_cache
from controlplane.observability import traceable_node
from controlplane.state import Stage


def _latency(state: Dict[str, Any]) -> float:
    return round((time.time() - float(state.get("started_at", time.time()))) * 1000, 1)


@traceable_node("finalize_block")
def finalize_block_node(state: Dict[str, Any]) -> Dict[str, Any]:
    cat = state.get("block_category", "policy")
    reason = state.get("block_reason", "The query was blocked by the input guardrail.")
    msg = (
        f"Request blocked by the input guardrail ({cat.replace('_', ' ')}).\n\n{reason}\n\n"
        "Rephrase your question without instructions that try to override the assistant's rules."
    )
    return {
        "stage": Stage.DONE,
        "stages_visited": [Stage.FINALIZE, Stage.DONE],
        "final_decision": "block",
        "final_answer": msg,
        "final_verdict_badges": ["BLOCKED", cat.upper().replace("_", " ")],
        "total_latency_ms": _latency(state),
    }


@traceable_node("finalize_cache")
def finalize_cache_node(state: Dict[str, Any]) -> Dict[str, Any]:
    ans = state.get("cached_answer", "")
    return {
        "stage": Stage.DONE,
        "stages_visited": [Stage.FINALIZE, Stage.DONE],
        "final_decision": "cache",
        "final_answer": ans,
        "answer": ans,
        "final_verdict_badges": ["CACHE HIT", f"sim {state.get('cache_similarity', 0):.2f}"],
        "total_latency_ms": _latency(state),
    }


@traceable_node("finalize_harmful")
def finalize_harmful_node(state: Dict[str, Any]) -> Dict[str, Any]:
    tox = state.get("toxicity", {}) or {}
    tox_lines = "\n".join(
        f"  - {k}: prob={v.get('prob')}, label={v.get('label')}"
        for k, v in tox.items()
        if isinstance(v, dict)
    )
    rules = state.get("violated_rules", []) or []
    report = state.get("resp_report") or ""
    msg = (
        "The generated answer was flagged as HARMFUL / NON-COMPLIANT and is not being delivered.\n\n"
        f"**Toxicity model outputs** (max={state.get('toxicity_max', 0):.2f}):\n{tox_lines or '  - n/a'}\n\n"
        f"**Rules / laws implicated:** {', '.join(rules) if rules else 'see analysis below'}\n\n"
        f"**Compliance analysis:**\n{report}"
    )
    return {
        "stage": Stage.DONE,
        "stages_visited": [Stage.FINALIZE, Stage.DONE],
        "final_decision": "harmful",
        "final_answer": msg,
        "final_verdict_badges": ["HARMFUL", f"toxicity {state.get('toxicity_max', 0):.2f}"]
        + [r[:40] for r in rules[:2]],
        "total_latency_ms": _latency(state),
    }


@traceable_node("finalize_safe")
def finalize_safe_node(state: Dict[str, Any]) -> Dict[str, Any]:
    ans = state.get("answer", "") or ""
    badges = ["SAFE"]
    if state.get("retry_count", 0):
        badges.append("HALLUCINATION-RETRIED")
    elif state.get("perf_verdict") == "hallucinated":
        badges.append("PERF-FLAGGED (retry skipped: latency budget)")
    if state.get("hitl_count", 0):
        badges.append("HITL-RESOLVED")
    rag = state.get("ragas_scores", {}) or {}
    if rag:
        badges.append(f"faithfulness {rag.get('faithfulness', 0):.2f}")

    # write-back only clean, first-pass-ish safe answers
    if (
        not state.get("hitl_count")
        and not state.get("retry_count")
        and state.get("perf_verdict") == "pass"
        and state.get("resp_status") == "safe"
        and ans
    ):
        try:
            get_cache().add(
                state.get("guarded_query") or state.get("original_query", ""),
                ans,
                meta={
                    "selected_kb": state.get("selected_kb"),
                    "model_used": state.get("model_used"),
                    "model_category": state.get("model_category"),
                },
            )
        except Exception:
            pass

    return {
        "stage": Stage.DONE,
        "stages_visited": [Stage.FINALIZE, Stage.DONE],
        "final_decision": "allow",
        "final_answer": ans,
        "final_verdict_badges": badges,
        "total_latency_ms": _latency(state),
    }
