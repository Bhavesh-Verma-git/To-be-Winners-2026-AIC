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
from controlplane.guardrails.intent import harmful_generation_request
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


# Queries that reference the *user's own* records/status - the knowledge base can
# never answer these without identifiers, so they always need a human round.
_PERSONAL_STATUS = re.compile(
    r"\b(has|have|did|was|when will|where is|what.?s the status of)\b[^?]{0,40}\bmy\b"
    r"|\bmy (leave request|leave application|request|application|order|refund|claim|case|ticket|"
    r"account|balance|appraisal|promotion|reimbursement|complaint|package|shipment)\b"
    r"|\b(am i|was i|has my \w+ been)\b[^?]{0,30}\b(approved|registered|enrolled|processed|"
    r"shortlisted|selected)\b",
    re.IGNORECASE,
)
# Broad / multi-part questions - a query rewrite that narrows the focus can pull
# sharper chunks, so these get ONE self-reflection retry before a human round.
_BROAD = re.compile(
    r"\b(summari[sz]e|list (all|every|each)|for each|every (major |key )?\w+|all (the )?\w+|"
    r"across all|complete (timeline|list|breakdown|summary|overview|guide)|detailed (timeline|breakdown|"
    r"overview|account|guide)|full (timeline|breakdown|list|account|picture|process|procedure|"
    r"workflow|guide|story|overview)|breakdown of|walk me through|the (whole|entire) \w+|"
    r"each (meeting|person|role|member|section|stage|phase|decision|step|choice|component)|"
    r"end to end|end-to-end|step[- ]by[- ]step)\b"
    r"|\b\w+:\s+\w+.*\band\b.*\band\b"          # "X: a, b, ... and c and d"  (enumerated multi-part)
    r"|\band\b[^.?!]{3,60}\band\b[^.?!]{3,60}\band\b",   # 3+ conjuncts in one clause
    re.IGNORECASE,
)

# Directionless asks with no concrete object - the answer can't be meaningfully
# "right", so always route to a human for the specifics.
_TOO_VAGUE = re.compile(
    r"^\s*(what should (we|i)\s+(do|decide|pick|choose)|help me (decide|choose|pick)|"
    r"which (one|option) (should|do) (we|i)|what do you (think|recommend|suggest)|"
    r"what.?s the best (choice|option|decision|approach)|what should be done|"
    r"any (advice|thoughts|recommendations?))\b[^?]{0,50}\??\s*$",
    re.IGNORECASE,
)

# A short question with a DANGLING topic reference that maps to several distinct
# answers in the KB ("the notice period" - resignation? probation?). Used ONLY as a
# tiebreaker when the model already said it can't answer -> HITL for the one detail
# that unlocks a real answer.
_AMBIGUOUS = re.compile(
    r"\b(the|my|our)\s+"
    r"(policy|process|procedure|rule|limit|cap|deadline|notice period|fee|charge|"
    r"timeline|duration|requirements?|eligibility|criteria|approval|discount|"
    r"bonus|benefits?|meeting|decision)\b",
    re.IGNORECASE,
)


def _q_ctx_overlap(query: str, rrf_chunks: List[Any]) -> float:
    """Jaccard-ish overlap between the query's content words and the retrieved
    chunks - i.e. did retrieval land on the right topic (regardless of whether
    the model chose to answer)."""
    qw = {w for w in re.findall(r"[a-z0-9]{4,}", (query or "").lower())}
    if not qw:
        return 0.0
    cw: set = set()
    for c in rrf_chunks[:5]:
        t = c.get("text", "") if isinstance(c, dict) else str(c)
        cw |= {w for w in re.findall(r"[a-z0-9]{4,}", t.lower())}
    return len(qw & cw) / len(qw)


def _needs_human(answer: str, ragas: Dict[str, float], rrf_chunks: List[Any], query: str = "") -> bool:
    if not rrf_chunks:
        return True
    cov = ragas.get("context_coverage", 1.0)
    rel = ragas.get("answer_relevancy", 1.0)
    low = answer.lower()
    said_cannot = any(m in low for m in _MISSING_INFO_MARKERS)

    # far-too-vague query -> a human must supply the specifics (RAGAS-independent)
    if _TOO_VAGUE.search((query or "").strip()):
        return True
    # personal-status queries -> the KB can never hold the user's own record;
    # trigger when the model said so OR retrieval clearly didn't cover the ask
    if _PERSONAL_STATUS.search(query or "") and (said_cannot or cov < 0.45):
        return True
    # short, on-topic but ambiguous question the model couldn't answer -> one
    # clarifying detail from the user will unlock a real answer
    if said_cannot and _AMBIGUOUS.search(query or "") and len((query or "").split()) <= 14 \
            and not _BROAD.search(query or ""):
        return True

    # the model explicitly said it cannot answer -> escalate unless RAGAS clearly
    # shows retrieval DID cover it (i.e. the model was just over-cautious). The
    # 0.6 heuristic fallback does NOT count as "clearly covered".
    if said_cannot and not (cov >= 0.7 and rel >= 0.7):
        return True
    return cov < 0.15 and rel < 0.30


_SUGG_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "for", "in", "on", "with", "how", "what", "which",
    "who", "when", "where", "why", "did", "do", "does", "is", "are", "was", "were", "give", "list",
    "summarize", "summarise", "describe", "explain", "tell", "me", "us", "each", "every", "all",
    "full", "detailed", "breakdown", "please", "across", "behind", "them", "that", "this", "their",
    "any", "some", "make", "state", "provide", "show", "one", "both",
}
_SUGG_BAD = re.compile(
    r"improved retrieval query|rewritten query|no preamble|the original (query|question)|"
    r"<=?\s*\d+\s*words|output only|grounding chunks|knowledge base\b.*query", re.IGNORECASE)


def _heuristic_suggestion(query: str) -> str:
    """No-LLM fallback: keep the salient noun-ish terms so the vector search
    focuses on the topic instead of the broad phrasing."""
    words = re.findall(r"[A-Za-z][A-Za-z\-']+", query.lower())
    keep = [w for w in words if w not in _SUGG_STOPWORDS and len(w) > 2]
    seen: List[str] = []
    for w in keep:
        if w not in seen:
            seen.append(w)
    return " ".join(seen[:12]) or query


def _suggestion(query: str, reasoning: str, drift: Dict[str, Any], unsupported: List[str]) -> str:
    hallucinated_ents = (drift or {}).get("entity_drift_results", {}).get("hallucinated_entities", [])
    fb = _heuristic_suggestion(query)
    ctx = (
        f"ORIGINAL QUESTION: {query}\n"
        f"PROBLEM: {reasoning}\n"
        f"unsupported claims: {unsupported[:5]}   fabricated entities: {hallucinated_ents[:8]}"
    )
    try:
        res = complete(
            "suggestion",
            [
                {"role": "system", "content": (
                    "You rewrite a failed RAG question into a short keyword-style search query "
                    "(3-12 words, no punctuation, no sentence) that will retrieve the right passages. "
                    "Reply with ONLY that query - nothing else.")},
                {"role": "user", "content": ctx},
            ],
            temperature=0.2,
            max_tokens=60,
        )
        text = (res.text or "").strip().strip('"').strip()
        text = text.splitlines()[0].strip() if text else ""
        if not text or _SUGG_BAD.search(text) or len(text.split()) > 20 or len(text) > 160:
            return fb
        return text
    except Exception:
        return fb


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

    answer_words = len(answer.split())
    said_cannot_now = any(m in answer.lower() for m in _MISSING_INFO_MARKERS)
    is_tox_route = state.get("selected_kb") == "toxicity_kb"
    # a confident, detailed answer that the purpose-trained model flags AND that
    # drifts from / contradicts the source (the model did NOT itself say "cannot
    # answer") -> the answer over-reached; reformulate the query and retry once.
    over_reached = (
        not said_cannot_now and not is_tox_route and answer_words >= 35
        and (
            (xgb_prob >= 0.70 and (faith < 0.55 or drift_score >= 0.20))
            or (drift_score >= 0.30 and len(factual_new) >= 2)
            or (faith < 0.35 and drift_score >= 0.15)
        )
    )
    hard_hallucination = (
        (xgb_prob >= 0.72 and (ragas_contradiction or fabricated_facts or relation_misstated))
        or fabricated_facts
        or (ragas_contradiction and relation_misstated)
        or (xgb_prob >= 0.93 and (ragas_verdict != "grounded" or drift_score >= 0.15))
        or over_reached
    )

    # ---- decision ----
    # A request to GENERATE hateful/discriminatory content is a responsibility
    # matter, not a grounding one - never send it to retry or HITL; let the
    # responsibility branch block it.
    if harmful_generation_request(query):
        return {
            "perf_verdict": "pass",
            "perf_reasoning": "harmful-content request - deferred to the responsibility branch.",
            "perf_score": 0.0,
            "detector_votes": votes,
        }

    said_cannot = any(m in answer.lower() for m in _MISSING_INFO_MARKERS)
    on_topic = _q_ctx_overlap(query, rrf_chunks) >= 0.55

    if hard_hallucination:
        verdict = "hallucinated"
        reasoning = (
            f"Hallucination: xgb={xgb_prob:.2f}, faithfulness={faith:.2f}, drift={drift_score:.2f}, "
            f"fabricated={hallucinated_ents[:5]}, relation_drift={relation_drift[:2]}, votes={votes}."
        )
    elif (said_cannot and on_topic and not _TOO_VAGUE.search((query or "").strip())
          and int(state.get("retry_count", 0)) < settings.max_hallucination_retries):
        # the model refused but the chunks ARE on-topic -> it was over-cautious.
        # One reformulated retry usually coaxes the real answer out (verdict EDIT).
        verdict = "hallucinated"
        reasoning = (
            f"Model declined although retrieval is on-topic (query<->context overlap "
            f"{_q_ctx_overlap(query, rrf_chunks):.2f}). Rewriting the query and retrying once."
        )
    elif _needs_human(answer, ragas, rrf_chunks, query):
        # A broad / multi-part on-topic query that retrieval couldn't cover is often
        # fixable by narrowing it: try ONE self-reflection retry (the agent rewrites
        # the query for sharper chunks) BEFORE falling back to a human. Personal-
        # record and directionless queries can't be reformulated - those go straight
        # to HITL.
        reformulable = (
            bool(rrf_chunks)                                   # something was retrieved, just off-target
            and bool(_BROAD.search(query or ""))               # a narrower rewrite could help
            and not _PERSONAL_STATUS.search(query or "")
            and not _TOO_VAGUE.search((query or "").strip())
        )
        if reformulable and int(state.get("retry_count", 0)) < settings.max_hallucination_retries:
            verdict = "hallucinated"
            reasoning = (
                "The answer was not grounded - retrieval did not cover this broad/multi-part "
                f"question (coverage={ragas.get('context_coverage', 0):.2f}, "
                f"relevancy={ragas.get('answer_relevancy', 0):.2f}). Rewriting the query for sharper chunks."
            )
        else:
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
        missing = _describe_missing(query, answer, ragas, state.get("selected_kb", ""))
        out["hitl_needed"] = True
        out["hitl_question"] = missing

    return out


def _describe_missing(query: str, answer: str, ragas: Dict[str, float], kb: str = "") -> str:
    """Ask for the ONE disambiguating detail that lets the KB answer - not personal
    identifiers the KB can't use. The reply is merged into the query and the whole
    pipeline re-runs."""
    topic = re.sub(r"\s+", " ", query).strip()[:160]
    q = query.lower()

    if kb == "hr_policy" or any(w in q for w in ("leave", "policy", "notice period", "allowance", "appraisal")):
        ask = ("**which specific policy topic** you need — e.g. *casual leave*, *sick leave*, "
               "*privilege leave*, *resignation notice period*, *probation rules*, or *travel allowance*")
    elif kb == "decision_support" or any(w in q for w in ("meeting", "decide", "team", "design", "remote")):
        ask = ("**which meeting or topic** — e.g. *the kickoff meeting*, *the conceptual design meeting*, "
               "*the detailed design meeting*, or a specific decision (*display*, *casing material*, "
               "*power source*, *target cost*)")
    elif kb == "customer_support" or any(w in q for w in ("order", "refund", "return", "delivery", "account")):
        ask = ("**which support topic** — e.g. *how to request a refund*, *how to return a damaged item*, "
               "*cancelling an order*, *tracking a package*, or *recovering a locked account*")
    elif kb == "internal_knowledge" or "azure" in q:
        ask = ("**which Azure App Service task** — e.g. *mapping a custom domain*, *binding a TLS/SSL "
               "certificate*, *configuring a staging slot*, *scaling*, or *deployment*")
    else:
        ask = ("**which specific topic, document or time-frame** you mean")

    return (
        f"I couldn't answer **“{topic}”** yet — it could mean several things and the retrieval "
        f"didn't land on one. Tell me {ask}.\n\n"
        "Your reply is added to the question and the whole pipeline re-runs."
    )
