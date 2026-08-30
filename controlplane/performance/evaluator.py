"""
Performance evaluator - fuses the 3 signals into one verdict.

Signals:  RAGAS (faithfulness / relevancy / coverage),  XGBoost hallucination prob,
          Entity drift (hallucinated entities + relation drift).

Verdict:
  "pass"         -> answer is grounded, continue
  "hallucinated" -> majority / hard signal says the answer is not grounded
                    => produce a RETRIEVAL-focused query rewrite (one `suggestion` LLM call)
  "need_human"   -> the context genuinely lacks the info the user asked for
                    => set a HITL question (no LLM call)

Aggregation reuses the "majority vote with safety bias" rules from
`master_router/performance_branch/performance_evaluator.py`.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from controlplane.config import settings
from controlplane.llm import complete

_MISSING_INFO_MARKERS = [
    "not covered in", "not found in", "could not find", "does not contain",
    "does not contain enough information", "not enough information",
    "no information", "not available in", "not mentioned in", "not specified in",
    "this specific information is not", "i don't have access", "i do not have access",
    "cannot determine from", "unable to answer", "was not found",
]


def _xgb_vote(prob: float) -> str:
    return "fail" if prob >= settings.xgb_hallucination_threshold else "pass"


def _ragas_vote(verdict: str, faith: float) -> str:
    if verdict == "hallucinated" or faith < settings.ragas_faithfulness_fail:
        return "fail"
    if verdict == "partially_grounded" or faith < settings.ragas_faithfulness_pass:
        return "warn"
    return "pass"


def _entity_vote(drift: Dict[str, Any]) -> str:
    v = (drift or {}).get("entity_drift_verdict", "pass")
    return {"fail": "fail", "warn": "warn"}.get(v, "pass")


def _needs_human(answer: str, ragas: Dict[str, float], rrf_chunks: List[Any]) -> bool:
    if not rrf_chunks:
        return True
    cov = ragas.get("context_coverage", 1.0)
    rel = ragas.get("answer_relevancy", 1.0)
    low = answer.lower()
    # the model explicitly said it cannot answer AND retrieval genuinely was thin
    if any(m in low for m in _MISSING_INFO_MARKERS) and (cov < 0.5 or rel < 0.5):
        return True
    return cov < 0.15 and rel < 0.30


def _suggestion(query: str, reasoning: str, drift: Dict[str, Any], unsupported: List[str]) -> str:
    hallucinated_ents = (drift or {}).get("entity_drift_results", {}).get("hallucinated_entities", [])
    ctx = (
        f"Original query: {query}\n"
        f"Why the answer was flagged: {reasoning}\n"
        f"Unsupported claims: {unsupported[:5]}\n"
        f"Fabricated entities: {hallucinated_ents[:8]}"
    )
    try:
        res = complete(
            "suggestion",
            [
                {
                    "role": "system",
                    "content": (
                        "The RAG answer was hallucinated. Produce ONE improved retrieval query "
                        "(<=40 words) that will pull the correct grounding chunks from the knowledge "
                        "base. Focus on precise terminology, entities, and the specific sub-topic. "
                        "Output only the rewritten query, no preamble."
                    ),
                },
                {"role": "user", "content": ctx},
            ],
            temperature=0.2,
            max_tokens=120,
        )
        text = res.text.strip().strip('"')
        return text or query
    except Exception:
        return query


def evaluate_performance(state: Dict[str, Any]) -> Dict[str, Any]:
    ragas = state.get("ragas_scores", {}) or {}
    ragas_verdict = state.get("ragas_verdict", "partially_grounded")
    xgb_prob = float(state.get("xgboost_prob") or 0.0)
    drift = state.get("entity_drift", {}) or {}
    answer = state.get("answer", "") or ""
    rrf_chunks = state.get("rrf_chunks", []) or []
    query = state.get("updated_query") or state.get("guarded_query") or state.get("original_query", "")

    faith = ragas.get("faithfulness", 1.0)
    dr = drift.get("entity_drift_results", {}) or {}
    drift_score = dr.get("drift_score", 0.0)
    hallucinated_ents = dr.get("hallucinated_entities", []) or []
    relation_drift = dr.get("relation_drift_pairs", []) or []
    entity_verdict = (drift or {}).get("entity_drift_verdict", "pass")

    votes = {
        "xgboost": "fail" if xgb_prob >= 0.72 else ("warn" if xgb_prob >= 0.55 else "pass"),
        "ragas": _ragas_vote(ragas_verdict, faith),
        "entity": _entity_vote(drift),
    }
    score = round(0.35 * faith + 0.45 * (1 - xgb_prob) + 0.20 * (1 - drift_score), 4)

    # A real RAG answer synthesised from context is NOT a hallucination just because
    # it isn't verbatim (low RAGAS faithfulness alone). We only flag when the purpose-
    # trained XGBoost model is confident AND a second signal corroborates, or when
    # multiple *factual* entities were demonstrably fabricated / a relationship misstated.
    factual_new = [e for e in hallucinated_ents
                   if any(c.isdigit() for c in e) or len(e.split()) >= 2]
    fabricated_facts = (
        entity_verdict == "fail" and drift_score >= 0.45 and len(factual_new) >= 3
    )
    relation_misstated = len(relation_drift) >= 2
    # "partially grounded / not verbatim" is fine; only an explicit contradiction counts
    ragas_contradiction = ragas_verdict == "hallucinated" or faith < 0.20

    hard_hallucination = (
        (xgb_prob >= 0.72 and (ragas_contradiction or fabricated_facts or relation_misstated))
        or fabricated_facts
        or (ragas_contradiction and relation_misstated)
        or (xgb_prob >= 0.93 and (ragas_verdict != "grounded" or drift_score >= 0.15))
    )

    # ---- decision ----
    if hard_hallucination:
        verdict = "hallucinated"
        reasoning = (
            f"Hallucination: xgb={xgb_prob:.2f}, faithfulness={faith:.2f}, drift={drift_score:.2f}, "
            f"fabricated={hallucinated_ents[:5]}, relation_drift={relation_drift[:2]}, votes={votes}."
        )
    elif _needs_human(answer, ragas, rrf_chunks):
        verdict = "need_human"
        reasoning = (
            "The retrieved context does not contain enough information to fully answer this "
            f"(coverage={ragas.get('context_coverage', 0):.2f}, relevancy={ragas.get('answer_relevancy', 0):.2f})."
        )
    else:
        verdict = "pass"
        soft = [k for k, v in votes.items() if v != "pass"]
        reasoning = (
            f"Answer accepted: xgb={xgb_prob:.2f}, faithfulness={faith:.2f}, drift={drift_score:.2f}, "
            f"score={score:.2f}." + (f" Soft flags: {soft}." if soft else "")
        )

    out: Dict[str, Any] = {
        "perf_verdict": verdict,
        "perf_reasoning": reasoning,
        "perf_score": score,
        "detector_votes": votes,
    }

    if verdict == "hallucinated" and int(state.get("retry_count", 0)) < settings.max_hallucination_retries:
        out["perf_suggestion"] = _suggestion(
            query, reasoning, drift, state.get("ragas_unsupported", []) or ragas.get("unsupported_claims", [])
        )
    elif verdict == "need_human":
        missing = _describe_missing(query, answer, ragas)
        out["hitl_needed"] = True
        out["hitl_question"] = missing

    return out


def _describe_missing(query: str, answer: str, ragas: Dict[str, float]) -> str:
    topic = re.sub(r"\s+", " ", query).strip()
    return (
        "I could not fully answer this from the knowledge base. To continue, please provide more "
        f"detail about: the specific case, product, policy section, time period, or identifiers "
        f"relevant to “{topic[:160]}”."
    )
