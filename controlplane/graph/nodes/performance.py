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

_RAGAS_TIMEOUT = float(os.getenv("CP_RAGAS_TIMEOUT_S", "3.8"))


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

    # RAGAS (Groq judge call) runs in a daemon thread so a slow judge can NEVER
    # hold the branch past _RAGAS_TIMEOUT - the executor's own shutdown() would
    # otherwise wait for it. (A stuck LLM call can't be killed; we just stop
    # waiting on it and fall back to the lexical heuristic scores.)
    ex = cf.ThreadPoolExecutor(max_workers=1)
    ragas_fut = None if is_retry else ex.submit(ragas_evaluate, query, answer, chunks)

    try:
        xgb_res = score_hallucination(context_str, answer, model_name, temperature)
    except Exception:
        xgb_res = {"hallucination_probability": 0.3, "risk_level": "LOW", "features": {}}
    try:
        drift_res = score_entity_drift(chunks or [answer[:400]], answer)
    except Exception:
        drift_res = {"entity_drift_verdict": "pass", "entity_drift_results": {}}

    # if the judge call times out, fall back to the no-LLM lexical heuristic (real
    # per-query scores) rather than a flat 0.6 - the flat value made the "cannot
    # answer" HITL gate misfire.
    from controlplane.performance.ragas_eval import _heuristic as _ragas_heuristic
    _hb = _ragas_heuristic(query, answer, context_str)
    if ragas_fut is not None:
        try:
            ragas_res = ragas_fut.result(timeout=_RAGAS_TIMEOUT)
        except Exception:
            ragas_res = {"scores": {k: _hb[k] for k in ("faithfulness", "answer_relevancy", "context_coverage")},
                         "verdict": _hb["verdict"], "unsupported_claims": []}
    else:
        ragas_res = {"scores": state.get("ragas_scores")
                     or {k: _hb[k] for k in ("faithfulness", "answer_relevancy", "context_coverage")},
                     "verdict": "partially_grounded", "unsupported_claims": []}
    ex.shutdown(wait=False)   # do NOT block on a still-running judge call

    if isinstance(ragas_res, dict):
        out["ragas_scores"] = ragas_res["scores"]
        out["ragas_verdict"] = ragas_res["verdict"]
        out["ragas_unsupported"] = ragas_res.get("unsupported_claims", [])
        if ragas_res.get("llm_call"):
            llm_calls.append({**ragas_res["llm_call"], "node": "ragas"})
    else:
        out["ragas_scores"] = {k: _hb[k] for k in ("faithfulness", "answer_relevancy", "context_coverage")}
        out["ragas_verdict"] = _hb["verdict"]
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
    if os.getenv("CP_DEBUG", "").lower() in {"1", "true", "yes"}:
        edr = drift_res.get("entity_drift_results", {})
        print(f"[cp:performance] verdict={verdict.get('perf_verdict')} xgb={out['xgboost_prob']:.2f} "
              f"ragas={out['ragas_scores']} drift={edr.get('drift_score')} "
              f"suggestion={str(verdict.get('perf_suggestion',''))[:80]!r} "
              f"branch_ms={out['perf_branch_ms']}", flush=True)
    if llm_calls:
        out["llm_calls"] = llm_calls
    return out
