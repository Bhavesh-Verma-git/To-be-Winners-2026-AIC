"""
RAGAS-style answer evaluation in ONE LiteLLM call.

The original `master_router/.../ragas_agent.py` does claim-decomposition with N+1
sequential LLM calls - far too slow for the <10 s budget. We keep the same
methodology (decompose -> check each claim against context -> aggregate) but push
it into a single structured-JSON prompt handled by the `judge` model category.

Output metrics: faithfulness, answer_relevancy, context_coverage
(+ unsupported_claims list for the dashboard / suggestion node).
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

from controlplane.config import settings
from controlplane.llm import complete_json

_SYSTEM = (
    "You are a rigorous RAG answer evaluator. You are given a QUESTION, the "
    "CONTEXT chunks that were retrieved, and an ANSWER produced from them. "
    "Judge ONLY against the context - outside knowledge does not count as support.\n"
    "Return a JSON object with exactly these keys:\n"
    '  "faithfulness": float 0-1  (fraction of answer claims fully supported by the context)\n'
    '  "answer_relevancy": float 0-1  (how directly the answer addresses the question)\n'
    '  "context_coverage": float 0-1  (how much of the answer-relevant info in the context the answer used)\n'
    '  "unsupported_claims": string[]  (short quotes of answer claims NOT supported by context; [] if none)\n'
    '  "verdict": "grounded" | "partially_grounded" | "hallucinated"\n'
    "Be strict but fair: a correct answer that simply says the info is not available is faithful."
)


def _clamp(x: Any, default: float = 0.5) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return default


def _heuristic(question: str, answer: str, ctx: str) -> Dict[str, Any]:
    """No-LLM fallback: lexical overlap of answer content words with the context."""
    import re

    def words(s):
        return {w for w in re.findall(r"[a-z0-9]{3,}", (s or "").lower())}

    a, c, q = words(answer), words(ctx), words(question)
    supported = len(a & c) / (len(a) or 1)
    relevant = len(a & q) / (len(q) or 1)
    faith = round(min(1.0, 0.35 + 0.75 * supported), 3)
    return {"faithfulness": faith, "answer_relevancy": round(min(1.0, 0.4 + relevant), 3),
            "context_coverage": round(min(1.0, supported + 0.2), 3),
            "unsupported_claims": [], "verdict": "grounded" if faith >= 0.6 else "partially_grounded"}


def ragas_evaluate(question: str, answer: str, context_chunks: List[str] | str) -> Dict[str, Any]:
    t0 = time.perf_counter()
    ctx = context_chunks if isinstance(context_chunks, str) else "\n\n".join(context_chunks or [])
    ctx = ctx[:6000]

    user = f"QUESTION:\n{question}\n\nCONTEXT:\n{ctx or '(no context retrieved)'}\n\nANSWER:\n{answer}"
    fb = _heuristic(question, answer, ctx)
    try:
        data, meta = complete_json(
            "judge",
            [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}],
            temperature=0.0, max_tokens=400, fallback=fb,
        )
        if data.get("_parse_error"):
            data = fb
        meta_rec = meta.as_call_record()
    except Exception:
        data, meta_rec = fb, None

    faith = _clamp(data.get("faithfulness"))
    rel = _clamp(data.get("answer_relevancy"))
    cov = _clamp(data.get("context_coverage"))
    verdict = str(data.get("verdict") or "").lower()
    if verdict not in {"grounded", "partially_grounded", "hallucinated"}:
        if faith >= settings.ragas_faithfulness_pass and rel >= 0.5:
            verdict = "grounded"
        elif faith < settings.ragas_faithfulness_fail:
            verdict = "hallucinated"
        else:
            verdict = "partially_grounded"

    return {
        "scores": {"faithfulness": faith, "answer_relevancy": rel, "context_coverage": cov},
        "verdict": verdict,
        "unsupported_claims": [str(c) for c in (data.get("unsupported_claims") or [])][:8],
        "reasoning": f"faithfulness={faith:.2f} relevancy={rel:.2f} coverage={cov:.2f}",
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        "llm_call": meta.as_call_record(),
    }
