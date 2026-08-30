"""
============================================================
  ControlPlane.ai — Shared Data Contract
  
  Every RAG agent (yours + your friend's) MUST return this exact
  shape. This means the Router, UI, and Trace panel never need
  to know how an agent works internally — just what it returns.
============================================================
"""
from typing import TypedDict, Optional


class RAGAgentOutput(TypedDict):
    """
    The universal output shape every agent must return.

    Fields
    ──────
    user_query   : The original question from the user (unchanged)
    rag_answer   : The final grounded answer text
    source       : The document source path (e.g. "documents/HR_Policy.pdf")
    source_url   : Clickable URL for citations (optional, agents that have it)
    agent_name   : Human-readable label: "Customer Support", "HR Policy", etc.
    route        : Internal route key: "customer_support", "hr_policy", etc.
    retrieved_n  : How many chunks were retrieved (for trace panel)
    has_code     : Whether the answer contains a code block (for Azure agent)
    error        : Set if something went wrong, otherwise None
    """
    user_query:  str
    rag_answer:  str
    source:      str
    source_url:  Optional[str]
    agent_name:  str
    route:       str
    retrieved_n: int
    has_code:    bool
    error:       Optional[str]


# ── Valid route keys ───────────────────────────────────────────
VALID_ROUTES = {
    "customer_support": "Customer Support",
    "hr_policy":        "HR Policy",
    "azure_docs":       "Azure App Service Docs",
    "toxicity":         "Toxicity Analysis",     # friend's agent
    "decision_support": "Decision Support",       # friend's agent
    "unknown":          "Unknown / Out of Scope",
}
