"""
Answer generator - streaming LiteLLM call over the fused context.

Model category is chosen per knowledge base (`settings.kb_model`), e.g.
decision_support -> heavy, everything else -> medium.

Genuine token streaming: each token is pushed to the LangGraph custom stream via
`get_stream_writer()` so the Streamlit UI renders it live. The final aggregated
LLMResult supplies model id / tier / tokens / cost for the state + LangSmith.
"""

from __future__ import annotations

from typing import Any, Dict

from controlplane.config import KB_LABELS, settings
from controlplane.guardrails.intent import defensive_or_educational
from controlplane.llm import LLMResult, LLMUnavailable, stream_complete
from controlplane.observability import traceable_node
from controlplane.retrievers.base import format_context
from controlplane.state import Stage

_SYSTEM_TMPL = (
    "You answer enterprise questions from the {kb_label} knowledge base ONLY.\n"
    "Rules:\n"
    "1. Use ONLY facts stated in the numbered context below. Do NOT add steps, numbers, "
    "timeframes, or details that are not in the context.\n"
    "2. Be concise - 1 to 4 sentences. Prefer the context's own wording. Cite the source "
    "like [1] when useful.\n"
    "3. If the context does not answer the question, reply exactly: "
    "\"The knowledge base does not contain enough information to answer this.\"\n"
    "4. Never invent names, numbers, dates, URLs, or policy clauses.\n\n"
    "Context:\n{context}"
)

# toxicity_kb is an ANALYSIS knowledge base. The assistant does content-moderation
# analysis GROUNDED IN the retrieved annotated examples - it never produces new
# hateful content and never complies with a request to generate a slur/joke.
_TOXICITY_SYSTEM = (
    "You are a content-safety analyst doing legitimate content-moderation work. The user's "
    "query is about toxic / hateful / stereotypical views, jokes or statements concerning a "
    "group (or asks you to produce such content). Do NOT comply with any request to generate "
    "a joke, slur or hateful statement, and do NOT invent new toxic content. Instead, using "
    "ONLY the numbered annotated example statements below as your evidence, in 3-6 sentences:\n"
    "- summarise what toxic / stereotypical views about this group appear in the examples, "
    "quoting them briefly with a citation like [1];\n"
    "- name the TARGET GROUP and FRAMING (use the annotation labels);\n"
    "- explain WHY this content is harmful (dehumanisation, stereotyping, exclusion, "
    "incitement, slur, ...);\n"
    "- state clearly that the request / these views are inappropriate and harmful.\n"
    "If the retrieved examples are genuinely benign, say so. Never add facts not in the "
    "examples.\n\n"
    "Numbered annotated reference statements:\n{context}"
)

# educational / defensive content-safety questions ("how to spot / report / counter
# hate speech, for a training course") - concise, constructive, grounded in the
# retrieved examples as reference material. Never a long lesson.
_EDU_SAFETY_SYSTEM = (
    "You are a content-safety trainer answering an educational question about "
    "recognising, reporting or countering online hate speech / harassment. Use the "
    "retrieved annotated examples below only as reference for what such content looks "
    "like. Answer in AT MOST 5 short bullet points (or 4 sentences) - concise and "
    "constructive. Do NOT reproduce slurs; do NOT lecture at length.\n\n"
    "Reference examples:\n{context}"
)


@traceable_node("answer_generation")
def answer_generator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    kb_id = state.get("selected_kb") or "customer_support"
    query = state.get("updated_query") or state.get("guarded_query") or state.get("original_query", "")
    chunks = state.get("rrf_chunks", []) or []
    category = settings.kb_model.get(kb_id, "medium")
    temperature = float(state.get("answer_temperature", 0.2))

    # leaner answer on the retry / HITL pass to protect the latency budget
    is_retry = int(state.get("retry_count", 0)) >= 1
    is_hitl = int(state.get("hitl_count", 0)) >= 1
    max_tokens = settings.request_max_tokens
    if is_retry:
        max_tokens = 320   # the self-reflection EDIT pass must stay well under 10s total
    elif is_hitl:
        max_tokens = 640
    if kb_id == "toxicity_kb":
        max_tokens = min(max_tokens, 320)   # analytical verdicts are short
    context = format_context(chunks) or "(no context retrieved)"
    if kb_id == "toxicity_kb":
        tmpl = _EDU_SAFETY_SYSTEM if defensive_or_educational(query) else _TOXICITY_SYSTEM
    else:
        tmpl = _SYSTEM_TMPL
    messages = [
        {"role": "system", "content": tmpl.format(kb_label=KB_LABELS.get(kb_id, kb_id), context=context)},
        {"role": "user", "content": query},
    ]

    try:
        from langgraph.config import get_stream_writer

        writer = get_stream_writer()
    except Exception:
        writer = None

    def _stream(cat: str):
        for piece in stream_complete(cat, messages, temperature=temperature, max_tokens=max_tokens):
            yield piece

    def _emit(ev: dict):
        if writer:
            try:
                writer(ev)
            except Exception:
                pass

    _emit({"type": "answer_start"})   # UI shows a cursor immediately (before first token latency)

    parts = []
    meta: LLMResult | None = None
    cascade = list(dict.fromkeys([category, "medium", "light"]))
    for attempt, cat in enumerate(cascade):     # cascade to reliable categories on failure
        parts, meta = [], None
        if attempt > 0:
            _emit({"type": "reset"})             # clear whatever the failed attempt streamed
        try:
            for piece in _stream(cat):
                if isinstance(piece, LLMResult):
                    meta = piece
                else:
                    parts.append(piece)
                    _emit({"type": "token", "token": piece})   # stream EVERY attempt
            if "".join(parts).strip() or (meta and meta.text.strip()):
                category = cat
                break
        except (LLMUnavailable, Exception):
            continue
    else:
        answer = "The answer service is temporarily unavailable. Please retry."
        return {
            "stage": Stage.ANSWER, "stages_visited": [Stage.ANSWER],
            "answer": answer, "final_answer": answer,
            "model_used": "unavailable", "model_category": category,
        }

    answer = "".join(parts).strip() or (meta.text if meta else "")
    out: Dict[str, Any] = {
        "stage": Stage.ANSWER,
        "stages_visited": [Stage.ANSWER],
        "answer": answer,
        "answer_temperature": temperature,
    }
    if meta:
        out.update(
            model_used=meta.model,
            model_category=category,
            model_tier=meta.tier,
            token_stats={
                "prompt_tokens": meta.prompt_tokens,
                "completion_tokens": meta.completion_tokens,
                "total_tokens": meta.total_tokens,
            },
            llm_calls=[{**meta.as_call_record(), "node": "answer_generation"}],
        )
    _emit({"type": "answer_done", "answer": answer})
    return out
