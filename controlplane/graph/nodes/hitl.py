"""
Human-in-the-loop nodes.

`hitl_interrupt` calls LangGraph's `interrupt()` with the question + context; the
graph pauses and the Streamlit UI shows exactly what is needed. On resume the
user's text is merged into `updated_query`, `hitl_count` is bumped to 1 (enforced
once), and the pipeline loops back to retrieval.
"""

from __future__ import annotations

from typing import Any, Dict

from controlplane.observability import traceable_node
from controlplane.state import Stage


@traceable_node("hitl_interrupt")
def hitl_interrupt_node(state: Dict[str, Any]) -> Dict[str, Any]:
    from langgraph.types import interrupt

    payload = {
        "question": state.get("hitl_question")
        or "More information is required to answer this reliably.",
        "original_query": state.get("original_query", ""),
        "draft_answer": state.get("answer", ""),
        "perf_reasoning": state.get("perf_reasoning", ""),
        "resp_status": state.get("resp_status", ""),
    }
    user_reply = interrupt(payload)  # <-- execution pauses here until resume

    reply = (user_reply or "").strip() if isinstance(user_reply, str) else str(user_reply or "")
    base = state.get("original_query", "")
    enriched = f"{base}\n\n[Additional information from user: {reply}]" if reply else base
    return {
        "stage": Stage.HITL,
        "stages_visited": [Stage.HITL],
        "hitl_count": int(state.get("hitl_count", 0)) + 1,
        "hitl_response": reply,
        "hitl_needed": False,
        "updated_query": enriched,
        "hitl_question": None,
    }
