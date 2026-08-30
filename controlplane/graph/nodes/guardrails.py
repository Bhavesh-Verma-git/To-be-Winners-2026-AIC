"""Guardrails node - injection/jailbreak block + PII masking. NO LLM. Runs first."""

from __future__ import annotations

from typing import Any, Dict

from controlplane.guardrails import mask_pii, scan_injection
from controlplane.observability import traceable_node
from controlplane.state import Stage


@traceable_node("guardrails")
def guardrails_node(state: Dict[str, Any]) -> Dict[str, Any]:
    query = state.get("original_query", "")

    verdict = scan_injection(query, use_embedding=True)
    if verdict.blocked:
        return {
            "stage": Stage.GUARDRAILS,
            "stages_visited": [Stage.GUARDRAILS],
            "blocked": True,
            "block_category": verdict.category,
            "block_reason": verdict.reason,
            "guardrail_flags": verdict.flags(),
            "final_decision": "block",
        }

    pii = mask_pii(query)
    flags = list(pii.flags())
    return {
        "stage": Stage.GUARDRAILS,
        "stages_visited": [Stage.GUARDRAILS],
        "blocked": False,
        "guarded_query": pii.text,
        "updated_query": pii.text,
        "guardrail_flags": flags,
        "guardrail_pii_spans": [{"type": t, "value": v} for t, v in pii.spans],
    }


def route_after_guardrails(state: Dict[str, Any]) -> str:
    return "blocked" if state.get("blocked") else "ok"
