"""
Bridge between the (async-friendly) LangGraph graph and the sync Streamlit UI.

`stream_run()` is a generator that:
  * drives `graph.stream(..., stream_mode=["custom", "updates"])`
  * yields ("token", str)         for each streamed answer token
  * yields ("node", node_name)    when a node finishes  (Tab 3 live view)
  * yields ("update", delta)      the raw state delta
  * yields ("interrupt", payload) when the graph pauses for HITL
  * yields ("final", state)       once, at the end
It also keeps `progress` (a dict passed by the caller) up to date so Tab 3 can
render the current stage even mid-stream.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, Iterator, Optional, Tuple

from controlplane.graph.build import build_graph
from controlplane.state import Stage, new_state

_NODE_TO_STAGE = {
    "guardrails": Stage.GUARDRAILS,
    "semantic_cache": Stage.CACHE,
    "rag_router": Stage.ROUTER,
    "retrieval": Stage.RETRIEVAL,
    "answer_generator": Stage.ANSWER,
    "performance": Stage.PERFORMANCE,
    "responsibility": Stage.RESPONSIBILITY,
    "aggregate": Stage.AGGREGATE,
    "hitl_interrupt": Stage.HITL,
    "finalize_block": Stage.FINALIZE,
    "finalize_cache": Stage.FINALIZE,
    "finalize_harmful": Stage.FINALIZE,
    "finalize_safe": Stage.FINALIZE,
}


def new_thread_id() -> str:
    return f"cp-{uuid.uuid4().hex[:12]}"


def stream_run(
    query: Optional[str],
    thread_id: str,
    *,
    resume: Optional[str] = None,
    progress: Optional[Dict[str, Any]] = None,
    forced_kb: Optional[str] = None,
) -> Iterator[Tuple[str, Any]]:
    graph = build_graph()
    run_id = str(uuid.uuid4())
    cfg = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 40,
        "run_id": run_id,
        "metadata": {"controlplane_run_id": run_id},
    }
    progress = progress if progress is not None else {}
    progress.setdefault("visited", [])
    progress["stage"] = Stage.START

    if resume is not None:
        from langgraph.types import Command

        graph_input: Any = Command(resume=resume)
    else:
        graph_input = new_state(query or "", thread_id, forced_kb=forced_kb)
        graph_input["langsmith_run_id"] = run_id

    last_state: Dict[str, Any] = {}
    _answer_stage_sent = False

    for mode, chunk in graph.stream(graph_input, cfg, stream_mode=["custom", "updates"]):
        if mode == "custom":
            if isinstance(chunk, dict) and chunk.get("type") == "token":
                if not _answer_stage_sent:
                    _answer_stage_sent = True
                    progress["stage"] = Stage.ANSWER
                    if Stage.ANSWER not in progress["visited"]:
                        progress["visited"].append(Stage.ANSWER)
                    yield ("node", "answer_generator")
                yield ("token", chunk["token"])
            elif isinstance(chunk, dict) and chunk.get("type") == "answer_start":
                if not _answer_stage_sent:
                    _answer_stage_sent = True
                    progress["stage"] = Stage.ANSWER
                    if Stage.ANSWER not in progress["visited"]:
                        progress["visited"].append(Stage.ANSWER)
                yield ("node", "answer_generator")
                yield ("answer_start", None)
            elif isinstance(chunk, dict) and chunk.get("type") == "reset":
                yield ("reset", None)
            elif isinstance(chunk, dict) and chunk.get("type") == "answer_done":
                yield ("answer_done", chunk.get("answer", ""))
        elif mode == "updates":
            for node, delta in (chunk or {}).items():
                if node == "__interrupt__":
                    payload = delta[0].value if isinstance(delta, (list, tuple)) and delta else delta
                    progress["stage"] = Stage.HITL
                    yield ("interrupt", payload)
                    return
                stage = _NODE_TO_STAGE.get(node, node)
                progress["stage"] = stage
                if stage not in progress["visited"]:
                    progress["visited"].append(stage)
                if isinstance(delta, dict):
                    last_state.update(delta)
                yield ("node", node)
                yield ("update", {node: delta})

    # pull the fully-merged state
    try:
        snap = graph.get_state(cfg)
        if snap and snap.values:
            last_state = dict(snap.values)
    except Exception:
        pass
    progress["stage"] = Stage.DONE
    yield ("final", last_state)
