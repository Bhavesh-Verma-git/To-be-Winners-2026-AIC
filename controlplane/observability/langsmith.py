"""
LangSmith wiring.

* `init_langsmith()` - sets the env LangChain/LangGraph read for auto-tracing and
  registers the LiteLLM langsmith callback. Safe no-op without a key.
* `traceable_node(name)` - decorator that makes each graph node its own span with
  timing, even when the caller isn't inside a LangChain runnable.
* `fetch_run_metrics(run_id)` - pulls the finished run tree back from LangSmith and
  flattens it into the per-node / per-model latency + token + cost rows the
  dashboard renders. Groq cost is forced to 0 here (single source of truth).
"""

from __future__ import annotations

import functools
import os
import time
from typing import Any, Callable, Dict, List, Optional

from controlplane.config import settings

_initialised = False


def init_langsmith() -> bool:
    global _initialised
    if _initialised:
        return settings.langsmith_enabled
    _initialised = True
    if not settings.langsmith_enabled:
        # make sure a placeholder key can't switch tracing on downstream
        for k in ("LANGCHAIN_TRACING_V2", "LANGSMITH_TRACING"):
            os.environ.pop(k, None)
        return False
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    # litellm's langsmith logger only reads LANGSMITH_API_KEY - mirror the LangChain one
    key = os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY")
    if key:
        os.environ["LANGSMITH_API_KEY"] = key
        os.environ.setdefault("LANGCHAIN_API_KEY", key)
    return True


def traceable_node(name: str) -> Callable:
    """Wrap a graph node with a per-node timing stamp.

    LangGraph already emits a LangSmith span per node when tracing is on, so we
    only add wall-clock timing here (surfaced in state['node_timings'] and the
    dashboard) - no extra span wrapper that could tangle the run tree.
    """

    def deco(fn: Callable) -> Callable:
        import asyncio

        if asyncio.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def a_inner(state, *a, **kw):
                t0 = time.perf_counter()
                return _stamp(await fn(state, *a, **kw), name, t0)

            return a_inner

        @functools.wraps(fn)
        def s_inner(state, *a, **kw):
            t0 = time.perf_counter()
            return _stamp(fn(state, *a, **kw), name, t0)

        return s_inner

    return deco


def _stamp(out: Any, name: str, t0: float) -> Any:
    ms = round((time.perf_counter() - t0) * 1000, 1)
    if isinstance(out, dict):
        timings = dict(out.get("node_timings", {}) or {})
        timings[name] = ms
        out["node_timings"] = timings
    return out


def fetch_run_metrics(run_id: Optional[str], project: Optional[str] = None) -> Dict[str, Any]:
    """Best-effort pull of a finished run tree. Returns {} if LangSmith is unavailable."""
    if not run_id or not settings.langsmith_enabled:
        return {}
    try:
        from langsmith import Client

        client = Client()
        runs = list(client.list_runs(project_name=project or settings.langsmith_project,
                                     run_ids=[run_id]))
        root = runs[0] if runs else None
        if root is None:
            return {}
        children = list(client.list_runs(project_name=project or settings.langsmith_project,
                                         trace_id=getattr(root, "trace_id", run_id)))
        nodes: List[Dict[str, Any]] = []
        models: List[Dict[str, Any]] = []
        total_cost = 0.0
        for r in children:
            dur = None
            if r.end_time and r.start_time:
                dur = (r.end_time - r.start_time).total_seconds() * 1000
            row = {"name": r.name, "type": r.run_type, "latency_ms": round(dur, 1) if dur else None}
            nodes.append(row)
            if r.run_type == "llm":
                model = (r.extra or {}).get("metadata", {}).get("ls_model_name") or r.name
                is_groq = "groq" in str(model).lower()
                cost = 0.0 if is_groq else float(getattr(r, "total_cost", 0) or 0)
                total_cost += cost
                models.append(
                    {
                        "model": model,
                        "latency_ms": row["latency_ms"],
                        "prompt_tokens": getattr(r, "prompt_tokens", None),
                        "completion_tokens": getattr(r, "completion_tokens", None),
                        "cost_usd": cost,
                        "provider": "groq" if is_groq else "other",
                    }
                )
        return {"nodes": nodes, "models": models, "total_cost_usd": round(total_cost, 6),
                "run_url": getattr(root, "url", None)}
    except Exception:
        return {}
