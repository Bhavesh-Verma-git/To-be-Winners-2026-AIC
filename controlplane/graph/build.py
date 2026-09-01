"""
LangGraph assembly.

    START
      -> guardrails --(blocked)--> finalize_block --> END
                    --(ok)------->  semantic_cache
      semantic_cache --(hit)----->  finalize_cache --> END
                     --(miss)---->  rag_router
      rag_router --> retrieval --> answer_generator
      answer_generator ==> performance   \\  (parallel superstep)
      answer_generator ==> responsibility //
      performance ---> aggregate
      responsibility -> aggregate
      aggregate --(harmful)--> finalize_harmful --> END
                --(retry)----> retrieval           (retry_count++ , once - same KB, rewritten query)
                --(hitl)-----> hitl_interrupt --> guardrails   (hitl_count++ , once - the user's info is
                                                                merged into the query and the WHOLE
                                                                pipeline re-runs from the start)
                --(safe)-----> finalize_safe --> END
"""

from __future__ import annotations

import uuid
from functools import lru_cache
from typing import Any, Dict, Optional

from langgraph.graph import END, START, StateGraph

from controlplane.graph.nodes.aggregate import aggregate_node, route_after_aggregate
from controlplane.graph.nodes.answer_generator import answer_generator_node
from controlplane.graph.nodes.finalize import (
    finalize_block_node,
    finalize_cache_node,
    finalize_harmful_node,
    finalize_safe_node,
)
from controlplane.graph.nodes.guardrails import guardrails_node, route_after_guardrails
from controlplane.graph.nodes.hitl import hitl_interrupt_node
from controlplane.graph.nodes.performance import performance_node
from controlplane.graph.nodes.rag_router import rag_router_node
from controlplane.graph.nodes.responsibility import responsibility_node
from controlplane.graph.nodes.retrieval import retrieval_node
from controlplane.graph.nodes.semantic_cache import route_after_cache, semantic_cache_node
from controlplane.observability import init_langsmith
from controlplane.state import ControlPlaneState, new_state


def _assemble() -> StateGraph:
    g = StateGraph(ControlPlaneState)

    g.add_node("guardrails", guardrails_node)
    g.add_node("semantic_cache", semantic_cache_node)
    g.add_node("rag_router", rag_router_node)
    g.add_node("retrieval", retrieval_node)
    g.add_node("answer_generator", answer_generator_node)
    g.add_node("performance", performance_node)
    g.add_node("responsibility", responsibility_node)
    g.add_node("aggregate", aggregate_node)
    g.add_node("hitl_interrupt", hitl_interrupt_node)
    g.add_node("finalize_block", finalize_block_node)
    g.add_node("finalize_cache", finalize_cache_node)
    g.add_node("finalize_harmful", finalize_harmful_node)
    g.add_node("finalize_safe", finalize_safe_node)

    g.add_edge(START, "guardrails")
    g.add_conditional_edges(
        "guardrails", route_after_guardrails,
        {"blocked": "finalize_block", "ok": "semantic_cache"},
    )
    g.add_conditional_edges(
        "semantic_cache", route_after_cache,
        {"hit": "finalize_cache", "miss": "rag_router"},
    )
    g.add_edge("rag_router", "retrieval")
    g.add_edge("retrieval", "answer_generator")
    g.add_edge("answer_generator", "performance")
    g.add_edge("answer_generator", "responsibility")
    g.add_edge("performance", "aggregate")
    g.add_edge("responsibility", "aggregate")
    g.add_conditional_edges(
        "aggregate", route_after_aggregate,
        {
            "harmful": "finalize_harmful",
            "retry": "retrieval",
            "hitl": "hitl_interrupt",
            "safe": "finalize_safe",
        },
    )
    # HITL merges the user's answer into the query and re-enters at guardrails, so
    # the enriched query runs the IDENTICAL full pipeline as a fresh query
    # (guardrails -> cache -> routing -> retrieval -> answer -> both branches ->
    # verdict). A toxic clarification therefore still reaches the content-safety KB.
    g.add_edge("hitl_interrupt", "guardrails")
    for term in ("finalize_block", "finalize_cache", "finalize_harmful", "finalize_safe"):
        g.add_edge(term, END)
    return g


@lru_cache(maxsize=1)
def build_graph():
    """Compile the graph once (with an in-process checkpointer for HITL interrupts)."""
    init_langsmith()
    from langgraph.checkpoint.memory import MemorySaver

    return _assemble().compile(checkpointer=MemorySaver())


def _config(thread_id: Optional[str], run_id: Optional[str] = None) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "configurable": {"thread_id": thread_id or f"cp-{uuid.uuid4().hex[:12]}"},
        "recursion_limit": 40,
    }
    if run_id:
        cfg["run_id"] = run_id
        cfg["metadata"] = {"controlplane_run_id": run_id}
    return cfg


def run_query_sync(
    query: str,
    *,
    conversation_id: str = "default",
    thread_id: Optional[str] = None,
    resume: Optional[str] = None,
    run_id: Optional[str] = None,
    forced_kb: Optional[str] = None,
) -> Dict[str, Any]:
    """Run to completion (or to a HITL interrupt) synchronously. If the graph paused
    for human input the returned dict contains an '__interrupt__' entry."""
    graph = build_graph()
    run_id = run_id or str(uuid.uuid4())
    cfg = _config(thread_id, run_id)
    if resume is not None:
        from langgraph.types import Command

        return graph.invoke(Command(resume=resume), cfg)
    state = new_state(query, conversation_id, forced_kb=forced_kb)
    state["langsmith_run_id"] = run_id
    return graph.invoke(state, cfg)


async def run_query(
    query: str,
    *,
    conversation_id: str = "default",
    thread_id: Optional[str] = None,
    resume: Optional[str] = None,
    forced_kb: Optional[str] = None,
) -> Dict[str, Any]:
    graph = build_graph()
    cfg = _config(thread_id)
    if resume is not None:
        from langgraph.types import Command

        return await graph.ainvoke(Command(resume=resume), cfg)
    return await graph.ainvoke(new_state(query, conversation_id, forced_kb=forced_kb), cfg)
