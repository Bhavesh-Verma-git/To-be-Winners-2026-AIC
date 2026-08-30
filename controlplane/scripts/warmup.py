"""
Warm every model + index so the first real query is not a cold start.

    python -m controlplane.scripts.warmup
"""

from __future__ import annotations

import time


def _timed(label, fn):
    t0 = time.time()
    try:
        fn()
        print(f"  [ok]   {label:32s} {time.time() - t0:6.1f}s")
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] {label:32s} {time.time() - t0:6.1f}s  {e}")


def main() -> None:
    print("Warming ControlPlane.ai ...")

    from controlplane.retrievers.registry import get_minilm, warm_all

    _timed("MiniLM embedder", lambda: get_minilm().encode(["warm"]))
    _timed("5 RAG knowledge bases", lambda: warm_all(verbose=False))

    from controlplane.performance.entity_drift import warmup as ed_warm
    from controlplane.performance.xgboost_infer import warmup as xgb_warm

    _timed("spaCy / entity drift", ed_warm)
    _timed("XGBoost + NLI features", xgb_warm)

    from controlplane.responsibility import get_responsibility_kb, get_toxicity_ensemble

    _timed("responsibility KB", lambda: get_responsibility_kb())
    _timed("toxicity ensemble", lambda: get_toxicity_ensemble().score_sync("warm up text"))

    from controlplane.graph import build_graph

    _timed("LangGraph compile", build_graph)

    from controlplane.llm import get_router

    _timed("LiteLLM router", get_router)

    # JIT-warm the whole graph path
    import os

    from controlplane.config import settings
    from controlplane.graph import run_query_sync

    if os.getenv("CP_WARMUP_FULL", "1") == "1":
        # a mock pass warms the code path with zero token cost
        prev = os.environ.get("CP_LLM_MOCK")
        os.environ["CP_LLM_MOCK"] = "1"
        try:
            _timed("mock graph pass", lambda: run_query_sync("How do I reset my password?", thread_id="warm-mock"))
        finally:
            os.environ.pop("CP_LLM_MOCK", None) if prev is None else os.environ.__setitem__("CP_LLM_MOCK", prev)

        # then ONE real pass so the first demo query isn't paying cold connection /
        # first-inference costs (streaming setup, RAGAS judge, real toxicity scoring)
        if settings.has_any_llm() and os.getenv("CP_WARMUP_REAL", "1") == "1":
            _timed("real graph pass", lambda: run_query_sync(
                "How do I return a damaged product?", thread_id="warm-real"))
            from controlplane.cache import get_cache
            get_cache().clear()   # don't leave the warmup query in the demo cache

    print("Done.")


if __name__ == "__main__":
    main()
