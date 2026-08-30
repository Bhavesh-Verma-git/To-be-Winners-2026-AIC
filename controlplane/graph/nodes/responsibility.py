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

    with cf.ThreadPoolExecutor(max_workers=3) as ex:
        # Score toxicity on BOTH the query AND the answer, take the max.
        # The harmful signal may live in the query (e.g. a slur) even when
        # the AI answer is clean educational prose.
        tox_query_fut = ex.submit(ens.score_sync, query)
        tox_answer_fut = ex.submit(ens.score_sync, answer) if answer else None
        retr = None
        if not retry_fast:
            try:
                retr = asyncio.run(kb.retrieve(f"{query}\n\n{answer}"))
            except Exception as e:  # noqa: BLE001
                retr = {"vector_chunks": [], "bm25_chunks": [], "graph_chunks": [],
                        "rrf_chunks": [], "meta": {"error": str(e)}}
        try:
            tox_q = tox_query_fut.result(timeout=8)
        except Exception:
            tox_q = {}
        tox_a = {}
        if tox_answer_fut is not None:
            try:
                tox_a = tox_answer_fut.result(timeout=8)
            except Exception:
                tox_a = {}
        # Merge: keep the higher toxicity_max across query and answer scores
        tox_max_q = float(tox_q.get("toxicity_max") or 0.0)
        tox_max_a = float(tox_a.get("toxicity_max") or 0.0)
        if tox_max_q >= tox_max_a:
            tox = tox_q
        else:
            tox = tox_a
        # Override toxicity_max with the combined maximum
        tox["toxicity_max"] = round(max(tox_max_q, tox_max_a), 4)

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
    return out
