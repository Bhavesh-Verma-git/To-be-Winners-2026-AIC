"""
Latency benchmark - asserts the p95 end-to-end wall-clock is under the budget.

LLM latency is either real (if keys are set) or a fixed injected budget
(`--llm-ms`, default 1200 ms per call in mock mode) so the retrieval / toxicity /
XGBoost / entity-drift / graph paths are measured for real.

    python -m controlplane.scripts.latency_bench
    python -m controlplane.scripts.latency_bench --n 20 --llm-ms 1500
"""

from __future__ import annotations

import argparse
import os
import statistics
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--llm-ms", type=int, default=1200, help="injected per-call LLM latency in mock mode")
    args = ap.parse_args()

    from controlplane.config import settings

    mock = not settings.has_any_llm()
    if mock:
        os.environ["CP_LLM_MOCK"] = "1"
        os.environ["CP_MOCK_LLM_MS"] = str(args.llm_ms)

    from controlplane.cache import get_cache
    from controlplane.prompts import DEMO_PROMPTS
    from controlplane.graph.build import run_query_sync
    from controlplane.scripts.warmup import main as warm

    print("Warming...")
    warm()
    get_cache().clear()  # measure cold retrieval every time

    seen = set()
    prompts = []
    for p in DEMO_PROMPTS:
        if p["kb"] in ("-",) or p["prompt"] in seen:
            continue
        seen.add(p["prompt"])
        prompts.append(p["prompt"])
        if len(prompts) >= args.n:
            break
    lat = []
    print(f"\nRunning {len(prompts)} queries (mock LLM={mock}, injected {args.llm_ms}ms/call)...")
    for i, q in enumerate(prompts, 1):
        t0 = time.time()
        st = run_query_sync(q, thread_id=f"bench-{i}")
        dt = time.time() - t0
        lat.append(dt)
        print(f"  {i:2d}. {dt:5.2f}s  [{st.get('final_decision', '?'):7s}] {q[:60]}")

    lat.sort()
    p50 = statistics.median(lat)
    p95 = lat[int(len(lat) * 0.95) - 1] if len(lat) > 1 else lat[-1]
    print(f"\n  p50={p50:.2f}s  p95={p95:.2f}s  max={max(lat):.2f}s  budget={settings.latency_budget_s:.0f}s")
    if p95 <= settings.latency_budget_s:
        print("  PASS - p95 within budget")
    else:
        print("  WARN - p95 exceeds budget; check the roberta-large-mnli / toxicity paths")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
