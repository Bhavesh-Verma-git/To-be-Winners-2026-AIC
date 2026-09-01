"""
RAG router - THE single main agent.

Selection order (first confident signal wins):
  0. manual override      - a KB explicitly picked in the sidebar (state["forced_kb"])
  1. keyword fast-path     - a strong keyword prior skips the LLM entirely
  2. LLM tool-calling      - the main agent calls one retrieve_* tool
  3. LLM constrained JSON  - fallback when the model can't do tool calls
  4. semantic KB probe     - embed the query, compare it to each KB's OWN content
                             (works even when the LLM refuses to engage, e.g. for
                             harmful / toxic queries). No keywords, no per-query rules.
  5. keyword prior / last-resort default

The node only *selects* the KB - the `retrieval` node executes the search.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List

from controlplane.config import KB_DESCRIPTIONS, KB_IDS, settings
from controlplane.llm import LLMUnavailable, complete, complete_json
from controlplane.observability import traceable_node
from controlplane.state import Stage

_DEBUG = os.getenv("CP_DEBUG", "").lower() in {"1", "true", "yes"}


def _dbg(msg: str) -> None:
    if _DEBUG:
        print(f"[cp:router] {msg}", flush=True)

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

# ---------------------------------------------------------------------------------------
# Semantic KB probe - the general fallback that makes toxic/harmful queries reach the
# content-safety KB WITHOUT any hardcoded keywords, groups or example queries.
#
# The LLM router refuses to engage with harmful queries ("I can't help with that"),
# so it never calls a retrieve_* tool for them. This probe does not need the LLM's
# cooperation: it embeds the query with the shared MiniLM model and compares it to
# each KB's OWN retrieved content (re-encoded so cosines are comparable across
# indexes built with different embedders). Whichever KB's material is most
# semantically similar to the query wins.
# ---------------------------------------------------------------------------------------
_SEM_MIN_SCORE = float(os.getenv("CP_ROUTER_SEM_MIN", "0.30"))
_SEM_MIN_MARGIN = float(os.getenv("CP_ROUTER_SEM_MARGIN", "0.06"))


def _semantic_kb_probe(query: str, top_chunks: int = 3) -> tuple[str | None, float, Dict[str, Any]]:
    try:
        from controlplane.retrievers import get_kb
        from controlplane.retrievers.registry import get_minilm
    except Exception:
        return None, 0.0, {}

    emb = get_minilm()
    if emb is None:
        return None, 0.0, {}
    try:
        qv = emb.encode([query], normalize_embeddings=True)[0]
    except Exception:
        return None, 0.0, {}

    scores: Dict[str, float] = {}
    for kb_id in KB_IDS:
        try:
            hits = get_kb(kb_id).vector_search(query, top_chunks)
            texts = [h.text[:400] for h in hits[:top_chunks] if getattr(h, "text", "")]
            if not texts:
                scores[kb_id] = 0.0
                continue
            hv = emb.encode(texts, normalize_embeddings=True)
            scores[kb_id] = float(max(hv @ qv))
        except Exception as exc:  # noqa: BLE001
            _dbg(f"semantic probe: {kb_id} failed - {exc}")
            scores[kb_id] = 0.0

    if not scores or max(scores.values()) <= 0.0:
        return None, 0.0, {}
    best = max(scores, key=scores.get)
    ordered = sorted(scores.values(), reverse=True)
    margin = ordered[0] - (ordered[1] if len(ordered) > 1 else 0.0)
    meta = {"scores": {k: round(v, 4) for k, v in scores.items()}, "margin": round(margin, 4)}
    confident = scores[best] >= _SEM_MIN_SCORE and margin >= _SEM_MIN_MARGIN
    return (best if confident else None), round(scores[best], 4), meta


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
    sem_meta: Dict[str, Any] = {}

    # 0) manual override - the user picked a KB in the sidebar. Use it directly,
    #    no routing, no LLM. Mode 2 (Retriever Selected).
    forced = (state.get("forced_kb") or "").strip()
    if forced and forced in KB_IDS:
        _dbg(f"manual override -> {forced!r}  (query={query[:80]!r})")
        return {
            "stage": Stage.ROUTER,
            "stages_visited": [Stage.ROUTER],
            "selected_kb": forced,
            "router_reason": "manual_selection",
            "router_confidence": 1.0,
        }

    # 1) strong keyword prior -> skip the LLM entirely (avoid an unnecessary call)
    kw_kb, kw_conf = _keyword_route(query)
    if kw_kb is not None and kw_conf >= _KEYWORD_SKIP_LLM:
        _dbg(f"keyword fast-path -> {kw_kb!r} conf={kw_conf}  (query={query[:80]!r})")
        return {
            "stage": Stage.ROUTER,
            "stages_visited": [Stage.ROUTER],
            "selected_kb": kw_kb,
            "router_reason": "keyword_fast_path",
            "router_confidence": round(kw_conf, 3),
        }

    # 2) tool-calling agent
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
        _dbg(f"tool-calling -> {kb!r} ({reason})  raw_text={((res.text or '')[:100])!r}")
    except (LLMUnavailable, Exception) as exc:
        _dbg(f"tool-calling raised: {exc}")

    # 3) constrained-JSON fallback
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
            _dbg(f"json fallback -> {cand!r}")
        except Exception as exc:
            _dbg(f"json fallback raised: {exc}")

    # 4) semantic KB probe - the LLM gave us nothing (or refused). Route by which
    #    KB's own content is most similar to the query. This is what makes harmful
    #    / toxic queries reach the content-safety KB without any hardcoded rules.
    if kb is None:
        sem_kb, sem_score, sem_meta = _semantic_kb_probe(query)
        _dbg(f"semantic probe -> {sem_kb!r} score={sem_score} meta={sem_meta}")
        if sem_kb is not None:
            kb, reason, conf = sem_kb, "semantic_probe", sem_score

    # 5) keyword prior, then a last-resort default (also seeded from the probe)
    kw_kb, kw_conf = _keyword_route(query)
    if kb is None and kw_kb is not None:
        kb, reason, conf = kw_kb, "keyword_fallback", kw_conf
    elif kb is None:
        # even a low-margin probe pick beats a blind customer_support default
        probe_scores = sem_meta.get("scores") or {}
        if probe_scores:
            kb = max(probe_scores, key=probe_scores.get)
            kb, reason, conf = kb, "semantic_probe_lowconf", float(probe_scores[kb])
        else:
            kb, reason, conf = "customer_support", "default_fallback", 0.2

    _dbg(f"FINAL route -> {kb!r} reason={reason} conf={round(conf, 3)}")
    out: Dict[str, Any] = {
        "stage": Stage.ROUTER,
        "stages_visited": [Stage.ROUTER],
        "selected_kb": kb,
        "router_reason": reason,
        "router_confidence": round(conf, 3),
        "router_semantic_scores": sem_meta.get("scores", {}),
    }
    if call_record:
        out["llm_calls"] = [{**call_record, "node": "rag_router"}]
    return out
