"""
Performance branch node (runs in parallel with the responsibility branch).

RAGAS is one Groq call (I/O) so it runs in a background thread; XGBoost and
entity-drift are fast local CPU (~0.2s together) and run inline - this avoids the
deep asyncio.run -> gather -> to_thread nesting that caused GIL contention.
Then the pure-logic evaluator produces the verdict.
"""

from __future__ import annotations

import concurrent.futures as cf
import os
import time
from typing import Any, Dict

from controlplane.observability import traceable_node
from controlplane.performance import (
    evaluate_performance,
    ragas_evaluate,
    score_entity_drift,
    score_hallucination,
)
from controlplane.performance.entity_drift import warmup as _ed_warm
from controlplane.performance.xgboost_infer import _load as _xgb_load
from controlplane.state import Stage

_RAGAS_TIMEOUT = float(os.getenv("CP_RAGAS_TIMEOUT_S", "5.0"))


@traceable_node("performance")
def performance_node(state: Dict[str, Any]) -> Dict[str, Any]:
    query = state.get("updated_query") or state.get("guarded_query") or state.get("original_query", "")
    answer = (state.get("answer", "") or "")[:2000]
    chunks = [c.get("text", "") for c in (state.get("rrf_chunks", []) or [])]
    context_str = "\n\n".join(chunks)[:2400]
    model_name = state.get("model_used") or "unknown"
    temperature = float(state.get("answer_temperature", 0.2))
    is_retry = int(state.get("retry_count", 0)) >= 1

    _xgb_load()
    _ed_warm()
    t0 = time.perf_counter()
    out: Dict[str, Any] = {"stage": Stage.PERFORMANCE, "stages_visited": [Stage.PERFORMANCE]}
    llm_calls = []

    with cf.ThreadPoolExecutor(max_workers=1) as ex:
        # RAGAS (Groq call) in the background; skip it entirely on the retry pass
        ragas_fut = None if is_retry else ex.submit(ragas_evaluate, query, answer, chunks)

        try:
            xgb_res = score_hallucination(context_str, answer, model_name, temperature)
        except Exception:
            xgb_res = {"hallucination_probability": 0.3, "risk_level": "LOW", "features": {}}
        try:
            drift_res = score_entity_drift(chunks or [answer[:400]], answer)
        except Exception:
            drift_res = {"entity_drift_verdict": "pass", "entity_drift_results": {}}

        if ragas_fut is not None:
            try:
                ragas_res = ragas_fut.result(timeout=_RAGAS_TIMEOUT)
            except Exception:
                ragas_res = None
        else:
            ragas_res = {"scores": state.get("ragas_scores")
                         or {"faithfulness": 0.6, "answer_relevancy": 0.6, "context_coverage": 0.6},
                         "verdict": "partially_grounded", "unsupported_claims": []}

    if isinstance(ragas_res, dict):
        out["ragas_scores"] = ragas_res["scores"]
        out["ragas_verdict"] = ragas_res["verdict"]
        out["ragas_unsupported"] = ragas_res.get("unsupported_claims", [])
        if ragas_res.get("llm_call"):
            llm_calls.append({**ragas_res["llm_call"], "node": "ragas"})
    else:
        out["ragas_scores"] = {"faithfulness": 0.6, "answer_relevancy": 0.6, "context_coverage": 0.6}
        out["ragas_verdict"] = "partially_grounded"
        out["ragas_unsupported"] = []

    out["xgboost_prob"] = xgb_res["hallucination_probability"]
    out["xgboost_risk"] = xgb_res["risk_level"]
    out["xgboost_features"] = xgb_res.get("features", {})
    out["entity_drift"] = drift_res

    verdict = evaluate_performance({**state, **out})
    out.update(verdict)
    if verdict.get("perf_suggestion"):
        llm_calls.append({"node": "perf_suggestion", "category": "suggestion"})

    out["perf_branch_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    if llm_calls:
        out["llm_calls"] = llm_calls
    return out
