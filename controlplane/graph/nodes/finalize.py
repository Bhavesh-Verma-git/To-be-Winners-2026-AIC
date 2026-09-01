"""
Terminal nodes - format the response for each outcome, write the cache on the
safe path, and stamp total latency.

  finalize_block   - guardrail rejection
  finalize_cache   - semantic-cache hit
  finalize_harmful - responsibility flagged the answer as unsafe
  finalize_safe    - normal answer (cache write happens here)
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict

from controlplane.cache import get_cache
from controlplane.observability import traceable_node
from controlplane.state import Stage


def _latency(state: Dict[str, Any]) -> float:
    return round((time.time() - float(state.get("started_at", time.time()))) * 1000, 1)


@traceable_node("finalize_block")
def finalize_block_node(state: Dict[str, Any]) -> Dict[str, Any]:
    cat = state.get("block_category", "policy")
    reason = state.get("block_reason", "The query was blocked by the input guardrail.")
    msg = (
        f"### Verdict: BLOCK\n\n"
        f"Request blocked by the input guardrail (**{cat.replace('_', ' ')}**).\n\n{reason}\n\n"
        "Rephrase your question without instructions that try to override the assistant's rules."
    )
    return {
        "stage": Stage.DONE,
        "stages_visited": [Stage.FINALIZE, Stage.DONE],
        "final_decision": "block",
        "final_verdict": "BLOCK",
        "final_answer": msg,
        "final_verdict_badges": ["BLOCK", cat.upper().replace("_", " ")],
        "total_latency_ms": _latency(state),
    }


@traceable_node("finalize_cache")
def finalize_cache_node(state: Dict[str, Any]) -> Dict[str, Any]:
    ans = state.get("cached_answer", "")
    return {
        "stage": Stage.DONE,
        "stages_visited": [Stage.FINALIZE, Stage.DONE],
        "final_decision": "cache",
        "final_verdict": "SAFE (cached)",
        "final_answer": ans,
        "answer": ans,
        "final_verdict_badges": ["SAFE", "CACHE HIT", f"sim {state.get('cache_similarity', 0):.2f}"],
        "total_latency_ms": _latency(state),
    }


_DEFAULT_RULES = [
    "EU AI Act (Reg. 2024/1689) — Article 5: prohibited AI practices",
    "NIST AI RMF (AI 100-1) — GOVERN 1.1 / MEASURE 2.11: fairness & harmful bias",
    "UN Universal Declaration of Human Rights — Article 1 (equal dignity)",
]


_TEMPLATE_LEAK = re.compile(
    r"then \d+.\d+ sentences|explaining your verdict|directly referencing what the candidate|"
    r"strict output format|do not deviate|\bOR\b\s*$|^- \*\*STATUS\*\*|"
    r"we need to evaluate the candidate|let me evaluate|the user query:|the ai[- ]generated answer:|"
    r"^(we|i) need to|^(let'?s|let me)\b", re.IGNORECASE)


def _clean_why(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text or _TEMPLATE_LEAK.search(text) or len(text) < 15:
        return ""
    return text[:400]


def _why_from_report(report: str) -> str:
    """Pull the 1-2 sentence rationale that follows the STATUS line in the
    Responsibility Agent's structured report (ignoring any leaked template text)."""
    if not report:
        return ""
    m = re.search(r"\*\*STATUS\*\*.*?(?:\n\n|\n)(.+?)(?:\n\n|\n---|\Z)", report, re.DOTALL)
    if m:
        cleaned = _clean_why(m.group(1))
        if cleaned:
            return cleaned
    for para in report.split("\n\n"):
        p = para.strip()
        if p and not p.startswith(("#", "-", "*", "═", "STATUS")):
            cleaned = _clean_why(p)
            if cleaned:
                return cleaned
    return ""


@traceable_node("finalize_harmful")
def finalize_harmful_node(state: Dict[str, Any]) -> Dict[str, Any]:
    tox = state.get("toxicity", {}) or {}
    tox_lines = "\n".join(
        f"- **{k}**: p={v.get('prob')}  → {v.get('label')}"
        for k, v in tox.items()
        if isinstance(v, dict)
    )
    rules = state.get("violated_rules", []) or _DEFAULT_RULES
    report = state.get("resp_report") or ""
    why = _why_from_report(report) or (
        "The request asks for hateful, discriminatory, or dehumanising content that targets a "
        "group defined by a protected characteristic (race, religion, gender, nationality, etc.). "
        "Producing or endorsing such content is prohibited."
    )
    rules_md = "\n".join(f"- {r}" for r in rules[:6])
    msg = (
        "### Verdict: BLOCK\n\n"
        "⚠️ **This request is inappropriate and harmful, so a normal answer was not generated.**\n\n"
        f"**Why:** {why}\n\n"
        f"**Laws / policies it violates:**\n{rules_md}\n\n"
        f"**Content-safety ensemble** (max toxicity {state.get('toxicity_max', 0):.2f}, "
        "scored on the query and the draft answer):\n"
        f"{tox_lines or '- n/a'}\n\n"
        "_The retrieved compliance clauses and every chunk pulled for this query are in the "
        "**Retrieval & Evidence** tab._"
    )
    return {
        "stage": Stage.DONE,
        "stages_visited": [Stage.FINALIZE, Stage.DONE],
        "final_decision": "harmful",
        "final_verdict": "BLOCK",
        "final_answer": msg,
        "violated_rules": rules,
        "final_verdict_badges": ["BLOCK", "HARMFUL", f"toxicity {state.get('toxicity_max', 0):.2f}"]
        + [r[:40] for r in rules[:2]],
        "total_latency_ms": _latency(state),
    }


_CANNOT_ANSWER_MARKERS = (
    "does not contain enough information", "not contain enough information",
    "no information", "not enough information", "cannot answer", "unable to answer",
    "temporarily unavailable",
)


@traceable_node("finalize_safe")
def finalize_safe_node(state: Dict[str, Any]) -> Dict[str, Any]:
    ans = state.get("answer", "") or ""
    retried = bool(state.get("retry_count", 0))
    hitl_done = bool(state.get("hitl_count", 0))
    original = state.get("original_answer") or ""
    if hitl_done:
        verdict = "HUMAN-IN-THE-LOOP"
    elif retried:
        verdict = "EDIT — self-reflection"
    else:
        verdict = "SAFE"
    badges = [verdict if verdict != "EDIT — self-reflection" else "EDIT (self-reflection)"]
    if verdict != "SAFE":
        badges.append("SAFE")
    if not retried and state.get("perf_verdict") == "hallucinated":
        badges.append("PERF-FLAGGED (retry skipped: latency budget)")
    rag = state.get("ragas_scores", {}) or {}
    if rag:
        badges.append(f"faithfulness {rag.get('faithfulness', 0):.2f}")

    # For the EDIT path show the before/after so the self-reflection is visible.
    final_answer = ans
    if retried and original and original.strip() != ans.strip():
        final_answer = (
            f"{ans}\n\n---\n"
            f"**↻ Self-reflection (verdict: EDIT)**\n\n"
            f"- **First draft was flagged:** {state.get('edit_reason', 'not fully grounded')[:220]}\n"
            f"- **Edit applied:** the agent rewrote the retrieval query "
            f"(`{str(state.get('perf_suggestion', ''))[:120]}`) and regenerated the answer shown above.\n"
            f"- **Original draft:** {original[:400]}"
        )

    # write-back: only clean, first-pass safe answers. Never cache a "cannot answer"
    # reply, and never cache the content-safety KB (its answers are analyses of
    # toxic material and must not be served to a differently-routed later query).
    ans_low = ans.lower()
    is_nonanswer = any(m in ans_low for m in _CANNOT_ANSWER_MARKERS)
    if (
        not state.get("hitl_count")
        and not state.get("retry_count")
        and state.get("perf_verdict") == "pass"
        and state.get("resp_status") == "safe"
        and state.get("selected_kb") != "toxicity_kb"
        and ans
        and not is_nonanswer
    ):
        try:
            get_cache().add(
                state.get("guarded_query") or state.get("original_query", ""),
                ans,
                meta={
                    "selected_kb": state.get("selected_kb"),
                    "model_used": state.get("model_used"),
                    "model_category": state.get("model_category"),
                },
            )
        except Exception:
            pass

    return {
        "stage": Stage.DONE,
        "stages_visited": [Stage.FINALIZE, Stage.DONE],
        "final_decision": "allow",
        "final_verdict": verdict,
        "final_answer": final_answer,
        "final_verdict_badges": badges,
        "total_latency_ms": _latency(state),
    }
