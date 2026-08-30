"""
Entity-drift detector - wraps the existing spaCy implementation
(`master_router/performance_branch/entity_drift_agent.py`). No LLM, local CPU.

Post-filter: spaCy tags list markers ("1)", "2)"), ordinals, small cardinals and
units as CARDINAL entities. Those are NOT fabricated facts, so we drop them from
`hallucinated_entities`, recompute the drift score, and re-derive the verdict.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Dict, List

from master_router.performance_branch.entity_drift_agent import EntityDriftAgent

_NUM_WORDS = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
              "first", "second", "third", "fourth", "fifth", "1st", "2nd", "3rd", "4th", "5th"}
# list markers / tiny cardinals only  (NOT years, counts, money - those are real facts)
_LIST_RE = re.compile(r"^\d{1,2}$")
_UNIT_RE = re.compile(
    r"^\d{1,3}(\s*[\-–]\s*\d{1,3}|\s*to\s*\d{1,3})?\s*"
    r"(day|days|hour|hours|week|weeks|month|months|business\s+days?|working\s+days?|"
    r"%|percent|minutes?|business\s+day)$",
    re.IGNORECASE,
)


def _is_trivial(ent: str) -> bool:
    e = ent.strip().lower().rstrip(").:,")
    if not e or e in _NUM_WORDS:
        return True
    if _LIST_RE.match(e):        # "1", "2", ... "12" list numbering
        return True
    if _UNIT_RE.match(e):        # "5-7 business days", "30 days", "10%"
        return True
    return False


@lru_cache(maxsize=1)
def _agent() -> EntityDriftAgent:
    a = EntityDriftAgent()
    a._initialize()
    return a


def warmup() -> None:
    _agent()


def _rederive(drift_score: float, overlap: float, halluc: List[str], relations: List[dict]) -> tuple[str, str]:
    if drift_score >= 0.40 and len(halluc) >= 2:
        return "fail", f"{len(halluc)} fabricated factual entities (drift {drift_score:.0%}): {halluc[:5]}"
    if len(relations) >= 2:
        return "fail", f"{len(relations)} entity relationships misstated: {relations}"
    if drift_score >= 0.25 and halluc:
        return "warn", f"possible fabricated entities (drift {drift_score:.0%}): {halluc[:5]}"
    if len(relations) >= 1:
        return "warn", f"a relationship between grounded entities may have changed: {relations}"
    return "pass", f"entities and relationships consistent with context (drift {drift_score:.0%})."


def score_entity_drift(retrieved_context: List[str] | str, answer: str) -> Dict[str, Any]:
    raw = _agent().score(retrieved_context=retrieved_context, rag_answer=answer)
    res = raw.get("entity_drift_results", {})

    halluc_all = res.get("hallucinated_entities", []) or []
    halluc = [e for e in halluc_all if not _is_trivial(e)]
    resp_ents = res.get("response_entities", []) or []
    n_resp = max(1, len([e for e in resp_ents if not _is_trivial(e)]))
    drift_score = round(len(halluc) / n_resp, 4)
    relations = res.get("relation_drift_pairs", []) or []
    overlap = res.get("entity_overlap_ratio", 1.0)

    verdict, reasoning = _rederive(drift_score, overlap, halluc, relations)
    res = {
        **res,
        "hallucinated_entities": halluc,
        "hallucinated_entities_raw": halluc_all,
        "drift_score": drift_score,
    }
    return {
        "entity_drift_results": res,
        "entity_drift_verdict": verdict,
        "entity_drift_reasoning": reasoning,
        "entity_drift_latency_ms": raw.get("entity_drift_latency_ms", 0.0),
    }
