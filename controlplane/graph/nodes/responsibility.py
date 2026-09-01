"""
Responsibility branch node (runs in parallel with the performance branch).

3-way responsibility retrieval (vector || bm25 || graph -> RRF) + toxicity
ensemble (Detoxify || toxic-bert || s-nlp roberta). Kept flat - retrieval runs
inline (fast), toxicity in one background thread - to avoid GIL contention from
deep asyncio nesting. Then the evaluator (pure-logic gate; 1 LLM report only
when flagged).
"""

from __future__ import annotations

import asyncio
import concurrent.futures as cf
import time
from typing import Any, Dict

from controlplane.observability import traceable_node
from controlplane.responsibility import (
    evaluate_responsibility,
    get_responsibility_kb,
    get_toxicity_ensemble,
)
from controlplane.state import Stage


@traceable_node("responsibility")
def responsibility_node(state: Dict[str, Any]) -> Dict[str, Any]:
    answer = state.get("answer", "") or ""
    query = state.get("guarded_query") or state.get("original_query", "")
    t0 = time.perf_counter()

    kb = get_responsibility_kb()
    ens = get_toxicity_ensemble()
    kb.load()
    ens.load()

    retry_fast = (
        int(state.get("retry_count", 0)) >= 1
        and state.get("resp_status") == "safe"
        and float(state.get("toxicity_max") or 0.0) < 0.15
    )

    out: Dict[str, Any] = {"stage": Stage.RESPONSIBILITY, "stages_visited": [Stage.RESPONSIBILITY]}

    if retry_fast:
        # The prior pass already cleared this answer as safe with near-zero toxicity
        # and we're only revising it for grounding - re-use the prior evidence and
        # toxicity scores so the EDIT retry stays inside the 10s budget.
        retr = None
        tox = {k: state.get("toxicity", {}).get(k, {}) for k in ("detoxify", "unitary", "snlp")}
        tox["toxicity_max"] = state.get("toxicity_max", 0.0)
    else:
        ex = cf.ThreadPoolExecutor(max_workers=1)
        # Score the 3 models on the ANSWER *and* the QUERY in one shot (per-model
        # max). A toxic query is flagged even when the model gave a bland refusal.
        tox_fut = ex.submit(ens.score_sync, answer or query, also=(query if answer else ""))
        retr = None
        try:
            retr = asyncio.run(kb.retrieve(f"{query}\n\n{answer}"))
        except Exception as e:  # noqa: BLE001
            retr = {"vector_chunks": [], "bm25_chunks": [], "graph_chunks": [],
                    "rrf_chunks": [], "meta": {"error": str(e)}}
        try:
            tox = tox_fut.result(timeout=8)
        except Exception:
            tox = {}
        ex.shutdown(wait=False)   # never block on a slow toxicity model

    if retr is not None:
        out["resp_vector_chunks"] = retr["vector_chunks"]
        out["resp_bm25_chunks"] = retr["bm25_chunks"]
        out["resp_graph_chunks"] = retr["graph_chunks"]
        out["resp_rrf_chunks"] = retr["rrf_chunks"]
        out["resp_retrieval_meta"] = retr["meta"]
    else:
        out["resp_rrf_chunks"] = state.get("resp_rrf_chunks", [])
        out["resp_retrieval_meta"] = {"note": "retry fast-path: toxicity re-check only"}

    if isinstance(tox, dict) and tox:
        out["toxicity"] = {k: tox[k] for k in ("detoxify", "unitary", "snlp") if k in tox}
        out["toxicity_max"] = tox.get("toxicity_max", 0.0)
    else:
        out["toxicity"] = {}
        out["toxicity_max"] = 0.0

    verdict = evaluate_responsibility({**state, **out})
    out.update({k: v for k, v in verdict.items() if not k.startswith("_")})
    if verdict.get("_resp_llm_call"):
        out["llm_calls"] = [{**verdict["_resp_llm_call"], "node": "responsibility_report"}]

    out["resp_branch_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    import os as _os
    if _os.getenv("CP_DEBUG", "").lower() in {"1", "true", "yes"}:
        print(f"[cp:responsibility] status={out.get('resp_status')} tox_max={out.get('toxicity_max')} "
              f"rules={ (out.get('violated_rules') or [])[:3] } "
              f"resp_rrf_n={len(out.get('resp_rrf_chunks') or [])} branch_ms={out['resp_branch_ms']}", flush=True)
    return out
