"""
============================================================
  ControlPlane.ai — Performance Fan-In Evaluator
  Performance Branch | Final Aggregation Node

  What it does:
  - Collects the verdicts from ALL 3 parallel evaluators:
      ① XGBoost Agent        (statistical pattern matching)
      ② RAGAS Agent          (LLM faithfulness judge)
      ③ Entity Drift Agent   (hard factual NER check)
  - Applies "Majority Vote with Safety Bias" to resolve conflicts.
  - Produces ONE final performance decision:
      "pass"  → Allow response to continue to user
      "retry" → Re-generate the RAG answer (max 1 retry)
      "hitl"  → Send to human review queue

  Conflict Resolution Rules:
  ┌─────────────────────────────────────────────────────────┐
  │ Entity Drift = fail           → ALWAYS retry (hard fact)│
  │ 2+ of 3 agents flag issue     → retry                   │
  │ RAGAS = fail + XGBoost = high → retry                   │
  │ Only 1 agent flags (uncertain)→ hitl                    │
  │ All 3 pass                    → pass                    │
  └─────────────────────────────────────────────────────────┘
============================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ── XGBoost hallucination probability threshold ────────────────
XGBOOST_HIGH_RISK_THRESHOLD = 0.60  # above this = hallucination signal


@dataclass
class PerformanceSignals:
    """
    Typed container for all 3 agent outputs.
    Makes the aggregation logic readable and testable.
    """
    # RAGAS
    ragas_verdict:          str   = "pass"
    ragas_faithfulness:     float = 1.0
    ragas_answer_relevancy: float = 1.0

    # XGBoost
    xgboost_hallucination_prob: float = 0.0
    xgboost_is_hallucination:   bool  = False
    xgboost_risk_level:         str   = "LOW"

    # Entity Drift
    entity_drift_verdict:       str   = "pass"
    drift_score:                float = 0.0
    hallucinated_entities:      list  = field(default_factory=list)

    @classmethod
    def from_state(cls, state: dict) -> "PerformanceSignals":
        """Build signals from LangGraph state."""
        ragas_scores   = state.get("ragas_scores", {})
        xgboost_scores = state.get("xgboost_scores", {})
        entity_results = state.get("entity_drift_results", {})

        return cls(
            # RAGAS
            ragas_verdict          = state.get("ragas_verdict", "pass"),
            ragas_faithfulness     = ragas_scores.get("faithfulness", 1.0),
            ragas_answer_relevancy = ragas_scores.get("answer_relevancy", 1.0),

            # XGBoost
            xgboost_hallucination_prob = xgboost_scores.get("hallucination_prob", 0.0),
            xgboost_is_hallucination   = xgboost_scores.get("is_hallucination", False),
            xgboost_risk_level         = xgboost_scores.get("risk_level", "LOW"),

            # Entity Drift
            entity_drift_verdict   = state.get("entity_drift_verdict", "pass"),
            drift_score            = entity_results.get("drift_score", 0.0),
            hallucinated_entities  = entity_results.get("hallucinated_entities", []),
        )


class PerformanceEvaluator:
    """
    The 'referee' that resolves conflicts between all 3 performance detectors.
    Pure logic — no LLM calls, no I/O.
    """

    def aggregate(self, signals: PerformanceSignals) -> dict[str, Any]:
        """
        Core aggregation logic with majority-vote safety bias.

        Returns:
            {
                "performance_evaluator_decision": "pass" | "retry" | "hitl",
                "performance_evaluator_reasoning": str,
                "performance_score": float,  # 0.0 = certain hallucination, 1.0 = clean
                "detector_votes": {
                    "xgboost": "pass" | "fail",
                    "ragas":   "pass" | "fail",
                    "entity":  "pass" | "warn" | "fail",
                },
            }
        """
        # ── Normalize each detector to a binary pass/fail vote ─
        xgb_vote     = self._xgboost_vote(signals)
        ragas_vote   = self._ragas_vote(signals)
        entity_vote  = self._entity_vote(signals)

        votes = {
            "xgboost": xgb_vote,
            "ragas":   ragas_vote,
            "entity":  entity_vote,
        }

        fail_count = sum(1 for v in votes.values() if v == "fail")
        warn_count = sum(1 for v in votes.values() if v == "warn")

        # ── Composite performance score ────────────────────────
        performance_score = self._compute_score(signals)

        # ── Decision Rules (Priority Ordered) ─────────────────
        decision, reasoning = self._decide(
            signals, votes, fail_count, warn_count, performance_score
        )

        logger.info(
            f"[PerformanceEvaluator] votes={votes} "
            f"fails={fail_count} warns={warn_count} "
            f"score={performance_score:.3f} → {decision}"
        )

        return {
            "performance_evaluator_decision":  decision,
            "performance_evaluator_reasoning": reasoning,
            "performance_score":               round(performance_score, 4),
            "detector_votes":                  votes,
        }

    # ── Individual vote derivers ───────────────────────────────

    def _xgboost_vote(self, s: PerformanceSignals) -> str:
        if s.xgboost_is_hallucination or s.xgboost_hallucination_prob >= XGBOOST_HIGH_RISK_THRESHOLD:
            return "fail"
        return "pass"

    def _ragas_vote(self, s: PerformanceSignals) -> str:
        if s.ragas_verdict == "fail":
            return "fail"
        if s.ragas_verdict == "uncertain":
            return "warn"
        return "pass"

    def _entity_vote(self, s: PerformanceSignals) -> str:
        if s.entity_drift_verdict == "fail":
            return "fail"
        if s.entity_drift_verdict == "warn":
            return "warn"
        return "pass"

    # ── Composite score ────────────────────────────────────────

    def _compute_score(self, s: PerformanceSignals) -> float:
        """
        Weighted composite. Faithfulness is weighted highest.
        Score of 1.0 = perfectly safe. Score of 0.0 = definite hallucination.
        """
        # Convert XGBoost probability to a "clean" score
        xgb_clean     = 1.0 - s.xgboost_hallucination_prob
        entity_clean  = 1.0 - s.drift_score
        ragas_clean   = s.ragas_faithfulness

        # Weights: RAGAS=40%, XGBoost=40%, Entity Drift=20%
        score = (0.40 * ragas_clean) + (0.40 * xgb_clean) + (0.20 * entity_clean)
        return max(0.0, min(1.0, score))  # clamp to [0,1]

    # ── Decision logic ─────────────────────────────────────────

    def _decide(
        self,
        s:                 PerformanceSignals,
        votes:             dict,
        fail_count:        int,
        warn_count:        int,
        performance_score: float,
    ) -> tuple[str, str]:
        """
        Priority-ordered decision rules:
        1. Entity Drift fail → always retry (hard factual lie = must fix)
        2. 2+ of 3 fail      → retry
        3. RAGAS fail + XGB high risk → retry
        4. Score very low    → retry
        5. 1 fail or 2 warns → hitl
        6. All pass          → pass
        """

        # Rule 1 — Hard factual entity lie. Always retry.
        if votes["entity"] == "fail":
            entities = s.hallucinated_entities[:5]  # Show max 5 for readability
            return "retry", (
                f"🚨 Entity Drift FAIL: LLM hallucinated factual entities: {entities}. "
                f"These were not in the source document. "
                f"XGBoost={votes['xgboost']} | RAGAS={votes['ragas']} | "
                f"Performance score={performance_score:.2f}. Re-generating."
            )

        # Rule 2 — Majority (2+) of 3 detectors flagged hallucination.
        if fail_count >= 2:
            return "retry", (
                f"⚠️ Majority vote: {fail_count}/3 detectors flagged hallucination. "
                f"XGBoost={votes['xgboost']} (prob={s.xgboost_hallucination_prob:.2f}) | "
                f"RAGAS={votes['ragas']} (faithfulness={s.ragas_faithfulness:.2f}) | "
                f"Entity={votes['entity']} (drift={s.drift_score:.2f}). "
                f"Score={performance_score:.2f}. Re-generating."
            )

        # Rule 3 — RAGAS and XGBoost agree on hallucination.
        if votes["ragas"] == "fail" and votes["xgboost"] == "fail":
            return "retry", (
                f"⚠️ RAGAS + XGBoost both flag hallucination. "
                f"Faithfulness={s.ragas_faithfulness:.2f} | "
                f"Hallucination prob={s.xgboost_hallucination_prob:.2f}. "
                f"Score={performance_score:.2f}. Re-generating."
            )

        # Rule 4 — Composite score is critically low.
        if performance_score < 0.35:
            return "retry", (
                f"⚠️ Composite performance score critically low: {performance_score:.2f}. "
                f"Votes: {votes}. Re-generating for safety."
            )

        # Rule 5 — Ambiguous: one agent failed or two warned. Send to human.
        if fail_count == 1 or warn_count >= 2:
            failed_agent = [k for k, v in votes.items() if v in ("fail", "warn")]
            return "hitl", (
                f"🔍 Borderline case — {failed_agent} raised concerns but majority passed. "
                f"Score={performance_score:.2f}. Routing to human review for final call."
            )

        # Rule 6 — Clean pass.
        return "pass", (
            f"✅ All performance detectors passed. "
            f"Faithfulness={s.ragas_faithfulness:.2f} | "
            f"XGBoost prob={s.xgboost_hallucination_prob:.2f} | "
            f"Entity drift={s.drift_score:.2f}. "
            f"Composite score={performance_score:.2f}."
        )


# ── LangGraph Fan-In Node ──────────────────────────────────────
_evaluator = PerformanceEvaluator()

def performance_fan_in_node(state: dict) -> dict:
    """
    LangGraph node — aggregates all 3 parallel performance evaluators.
    Reads:  state['ragas_scores'], state['ragas_verdict'],
            state['xgboost_scores'], state['entity_drift_results'],
            state['entity_drift_verdict']
    Writes: state['performance_evaluator_decision'],
            state['performance_evaluator_reasoning'],
            state['performance_score'], state['detector_votes']
    """
    signals = PerformanceSignals.from_state(state)
    result  = _evaluator.aggregate(signals)
    return {**state, **result}


# ── CLI smoke test ─────────────────────────────────────────────
if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)

    ev = PerformanceEvaluator()

    print("\n" + "="*60)
    print("TEST 1: All clean (expected: PASS)")
    print("="*60)
    sigs = PerformanceSignals(
        ragas_verdict="pass", ragas_faithfulness=0.92,
        xgboost_hallucination_prob=0.08, xgboost_is_hallucination=False, xgboost_risk_level="LOW",
        entity_drift_verdict="pass", drift_score=0.05, hallucinated_entities=[],
    )
    print(json.dumps(ev.aggregate(sigs), indent=2))

    print("\n" + "="*60)
    print("TEST 2: Entity drift fails (expected: RETRY)")
    print("="*60)
    sigs = PerformanceSignals(
        ragas_verdict="uncertain", ragas_faithfulness=0.55,
        xgboost_hallucination_prob=0.45, xgboost_is_hallucination=False, xgboost_risk_level="MODERATE",
        entity_drift_verdict="fail", drift_score=0.65,
        hallucinated_entities=["jane doe", "san francisco", "1990"],
    )
    print(json.dumps(ev.aggregate(sigs), indent=2))

    print("\n" + "="*60)
    print("TEST 3: Only 1 agent flags (expected: HITL)")
    print("="*60)
    sigs = PerformanceSignals(
        ragas_verdict="fail", ragas_faithfulness=0.30,
        xgboost_hallucination_prob=0.20, xgboost_is_hallucination=False, xgboost_risk_level="LOW",
        entity_drift_verdict="pass", drift_score=0.05, hallucinated_entities=[],
    )
    print(json.dumps(ev.aggregate(sigs), indent=2))
