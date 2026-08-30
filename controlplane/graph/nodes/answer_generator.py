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

# toxicity_kb is an ANALYSIS knowledge base - the assistant must explain why a
# statement is/ isn't harmful, not refuse. This is legitimate content moderation.
_TOXICITY_SYSTEM = (
    "You are a content-safety analyst. Analyse the STATEMENT OR PHRASE the user is asking "
    "about (the quoted text, or the language/expression they name - NOT their question "
    "itself). This is legitimate content-moderation work: do NOT refuse, do NOT repeat "
    "slurs gratuitously.\n"
    "Use the annotated example statements below as reference. In 2-5 sentences give:\n"
    "- verdict: is that statement/phrase toxic / hate speech / harmful, and how severe;\n"
    "- the TARGET GROUP and FRAMING (use the annotation labels);\n"
    "- WHY (dehumanisation, stereotyping, exclusion, incitement, slur, ...).\n"
    "If the phrase is genuinely benign, say so. Cite reference examples like [1].\n\n"
    "Annotated reference statements:\n{context}"
)


@traceable_node("answer_generation")
def answer_generator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    kb_id = state.get("selected_kb") or "customer_support"
    query = state.get("updated_query") or state.get("guarded_query") or state.get("original_query", "")
    chunks = state.get("rrf_chunks", []) or []
    category = settings.kb_model.get(kb_id, "medium")
    temperature = float(state.get("answer_temperature", 0.2))

    # leaner answer on the retry pass to protect the latency budget
    is_retry = int(state.get("retry_count", 0)) >= 1 or int(state.get("hitl_count", 0)) >= 1
    max_tokens = 640 if is_retry else settings.request_max_tokens
    context = format_context(chunks) or "(no context retrieved)"
    tmpl = _TOXICITY_SYSTEM if kb_id == "toxicity_kb" else _SYSTEM_TMPL
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

    parts = []
    meta: LLMResult | None = None
    cascade = list(dict.fromkeys([category, "medium", "light"]))
    for attempt, cat in enumerate(cascade):     # cascade to reliable categories on failure
        parts, meta = [], None
        try:
            for piece in _stream(cat):
                if isinstance(piece, LLMResult):
                    meta = piece
                else:
                    parts.append(piece)
                    if writer and attempt == 0:  # only stream the first attempt to the UI
                        try:
                            writer({"type": "token", "token": piece})
                        except Exception:
                            pass
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
    if writer:
        try:
            writer({"type": "answer_done", "answer": answer})
        except Exception:
            pass
    return out
