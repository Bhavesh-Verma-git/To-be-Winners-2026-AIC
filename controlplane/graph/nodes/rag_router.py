"""
RAG router - THE single main agent.

One LiteLLM call decides which knowledge base answers the query. Primary path is
native tool-calling over 5 retriever tools; if the model/deployment doesn't
support tools we fall back to a constrained-JSON classification. A keyword prior
is the last resort so routing never hard-fails.

The node only *selects* the KB - the `retrieval` node executes the tool.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from controlplane.config import KB_DESCRIPTIONS, KB_IDS, settings
from controlplane.llm import LLMUnavailable, complete, complete_json
from controlplane.observability import traceable_node
from controlplane.state import Stage

_TOOLS: List[dict] = [
    {
        "type": "function",
        "function": {
            "name": f"retrieve_{kb}",
            "description": KB_DESCRIPTIONS[kb],
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "search query for this KB"}},
                "required": ["query"],
            },
        },
    }
    for kb in KB_IDS
]

_KEYWORDS = {
    "customer_support": ["refund", "order", "cancel", "delivery", "shipping", "shipped", "return",
                         "returned", "invoice", "billing", "charged", "account", "password", "login",
                         "log in", "payment", "subscription", "track", "tracking", "package", "agent",
                         "complaint"],
    "hr_policy": ["leave", "leaves", "sick", "casual", "privilege", "salary", "appraisal", "promotion",
                  "attendance", "dress code", "resignation", "resign", "notice period", "kespl",
                  "probation", "probationer", "allowance", "working hours", "holiday", "employee",
                  "termination", "sanctioning authority", "late"],
    "internal_knowledge": ["azure", "app service", "deployment", "slot", "custom domain", "tls", "ssl",
                           "certificate", "vnet", "az webapp", "cli", "scale", "autoscale", "kudu",
                           "web app", "managed identity", "appsettings", "application settings",
                           "staging", "github actions", "ssh", "hosting plan"],
    "toxicity_kb": ["toxic", "toxicity", "offensive", "hate speech", "slur", "this statement",
                    "the statement", "abusive", "stereotype", "stereotyping", "harass", "bigot",
                    "inferior", "discriminat", "target group", "framing", "belong here",
                    "fall under", "is calling", "is saying", "is 'this", "analyze whether",
                    "classify the", "rate the", "dehuman", "bigotry", "prejudice", "racist",
                    "sexist", "xenophob", "go back to"],
    "decision_support": ["meeting", "meetings", "decision", "decide", "decided", "target cost",
                         "remote control", "lcd", "led", "prototype", "design team", "marketing",
                         "industrial designer", "battery", "solar", "product design", "demographic",
                         "casing", "button layout", "budget", "cfo", "tooling"],
}

_SYSTEM = (
    "You route an enterprise user query to exactly ONE knowledge base by calling the matching "
    "retrieve_* tool with a focused search query. If none of the knowledge bases fit, call no tool.\n\n"
    + "\n".join(f"- {kb}: {KB_DESCRIPTIONS[kb]}" for kb in KB_IDS)
)


def _keyword_route(query: str) -> tuple[str | None, float]:
    q = query.lower()
    scored = {kb: sum(1 for kw in kws if kw in q) for kb, kws in _KEYWORDS.items()}
    best = max(scored, key=scored.get)
    hits = scored[best]
    runner_up = sorted(scored.values())[-2] if len(scored) > 1 else 0
    if hits == 0:
        return None, 0.0
    # confident only if clearly ahead of the runner-up
    conf = min(0.92, 0.45 + 0.16 * hits - 0.08 * runner_up)
    return best, round(conf, 3)


def _from_tool_calls(raw) -> tuple[str | None, str, float]:
    try:
        tcs = raw.choices[0].message.tool_calls or []
    except Exception:
        return None, "", 0.0
    for tc in tcs:
        name = getattr(getattr(tc, "function", None), "name", "") or ""
        if name.startswith("retrieve_"):
            kb = name[len("retrieve_"):]
            if kb in KB_IDS:
                return kb, "tool_call", 0.85
    return None, "", 0.0


_KEYWORD_SKIP_LLM = float(__import__("os").getenv("CP_ROUTER_KEYWORD_SKIP", "0.62"))


@traceable_node("rag_router")
def rag_router_node(state: Dict[str, Any]) -> Dict[str, Any]:
    query = state.get("updated_query") or state.get("guarded_query") or state.get("original_query", "")
    kb, reason, conf = None, "", 0.0
    call_record = None

    # 0) strong keyword prior -> skip the LLM entirely (avoid an unnecessary call)
    kw_kb, kw_conf = _keyword_route(query)
    if kw_kb is not None and kw_conf >= _KEYWORD_SKIP_LLM:
        return {
            "stage": Stage.ROUTER,
            "stages_visited": [Stage.ROUTER],
            "selected_kb": kw_kb,
            "router_reason": "keyword_fast_path",
            "router_confidence": round(kw_conf, 3),
        }

    # 1) tool-calling agent
    try:
        res = complete(
            "main_agent",
            [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": query}],
            temperature=0.0,
            tools=_TOOLS,
            tool_choice="auto",
            max_tokens=256,
        )
        call_record = res.as_call_record()
        kb, reason, conf = _from_tool_calls(res.raw)
        if kb is None and res.text:
            m = re.search(r"retrieve_(\w+)", res.text)
            if m and m.group(1) in KB_IDS:
                kb, reason, conf = m.group(1), "text_mention", 0.6
    except (LLMUnavailable, Exception):
        pass

    # 2) constrained-JSON fallback
    if kb is None:
        try:
            data, meta = complete_json(
                "main_agent",
                [
                    {"role": "system", "content": _SYSTEM + "\nRespond as JSON: "
                     '{"knowledge_base": "<one of the ids or none>", "confidence": 0-1, "reason": "..."}'},
                    {"role": "user", "content": query},
                ],
                temperature=0.0,
                fallback={"knowledge_base": "none", "confidence": 0.0, "reason": "fallback"},
            )
            call_record = call_record or meta.as_call_record()
            cand = str(data.get("knowledge_base", "none")).strip()
            if cand in KB_IDS:
                kb, reason, conf = cand, str(data.get("reason", "json"))[:200], float(data.get("confidence", 0.5) or 0.5)
        except Exception:
            pass

    # 3) keyword prior (or to disambiguate a low-confidence LLM pick)
    kw_kb, kw_conf = _keyword_route(query)
    if kb is None and kw_kb is not None:
        kb, reason, conf = kw_kb, "keyword_fallback", kw_conf
    elif kb is None:
        kb, reason, conf = "customer_support", "default_fallback", 0.2

    out: Dict[str, Any] = {
        "stage": Stage.ROUTER,
        "stages_visited": [Stage.ROUTER],
        "selected_kb": kb,
        "router_reason": reason,
        "router_confidence": round(conf, 3),
    }
    if call_record:
        out["llm_calls"] = [{**call_record, "node": "rag_router"}]
    return out
