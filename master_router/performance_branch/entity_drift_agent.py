"""
============================================================
  ControlPlane.ai — Entity Drift Agent
  Performance Branch | Parallel Eval Node

  What it does:
  - Uses spaCy NER to identify all Named Entities (people,
    places, numbers, organizations, dates) in both the
    retrieved context and the generated answer.
  - Flags any entity in the answer that was NOT in the context
    as a "hallucinated entity" — a hard factual lie.
  - Also tracks "drift" across multi-turn conversations
    (did the entity set suddenly change between turns?).

  This is the FASTEST of the 3 performance detectors because
  it uses no LLM API calls at all. 100% local CPU.

  Also checks RELATION drift: two entities can both be present
  in the context (zero "hallucinated_entities") while the
  relationship between them has been swapped, e.g.
    context: "Tesla's CEO is Elon Musk"
    answer:  "Tesla was founded by Elon Musk"
  Pure entity-set overlap misses this. We catch it for free by
  reusing the dependency parse spaCy already computes for NER
  (no extra model call, no extra latency) to pull a coarse
  (entity, relation-label, entity) triple per sentence and diff
  it against the same triples extracted from the context.

  Output shape:
  {
    "entity_drift_results": {
      "context_entities":       list[str],
      "response_entities":      list[str],
      "hallucinated_entities":  list[str],   # NEW entities not in context
      "drift_score":            float,       # 0.0 = clean, 1.0 = all new
      "entity_overlap_ratio":   float,       # 1.0 = perfectly grounded
      "multi_turn_drift_score": float,       # vs previous turn
      "relation_drift_pairs":   list[dict],  # entity pairs whose relation changed
      "relation_drift_score":   float,       # 0.0 = clean, 1.0 = all relations changed
    },
    "entity_drift_verdict":   "pass" | "warn" | "fail",
    "entity_drift_reasoning": str,
    "entity_drift_latency_ms": float,
  }
============================================================
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# ── Thresholds ─────────────────────────────────────────────────
DRIFT_SCORE_FAIL_THRESHOLD = 0.40  # >40% new entities → fail
DRIFT_SCORE_WARN_THRESHOLD = 0.20  # >20% new entities → warn
OVERLAP_FAIL_THRESHOLD     = 0.50  # <50% entity overlap → suspect
RELATION_DRIFT_FAIL_COUNT  = 2     # >=2 swapped relation pairs → fail
RELATION_DRIFT_WARN_COUNT  = 1     # >=1 swapped relation pair  → warn


class EntityDriftAgent:
    """
    Synchronous entity drift detector using spaCy.
    Zero external API calls. Pure local NER.
    """

    def __init__(self) -> None:
        self._nlp = None
        self._initialized = False

    def _initialize(self) -> None:
        """Lazy-load spaCy model once."""
        if self._initialized:
            return
        try:
            import spacy
            self._nlp = spacy.load("en_core_web_sm")
        except OSError:
            from spacy.cli import download
            logger.info("[EntityDriftAgent] Downloading en_core_web_sm...")
            download("en_core_web_sm")
            import spacy
            self._nlp = spacy.load("en_core_web_sm")
        self._initialized = True
        logger.info("[EntityDriftAgent] spaCy NER loaded.")

    def _parse(self, text: str):
        """
        Runs the text through spaCy once. Truncates to 100k chars to
        avoid spaCy limits on huge contexts. Returns None for empty input.
        """
        if not text or not isinstance(text, str):
            return None
        return self._nlp(text[:100_000])

    @staticmethod
    def _entities_from_doc(doc) -> set[str]:
        if doc is None:
            return set()
        return {ent.text.lower().strip() for ent in doc.ents if ent.text.strip()}

    def _extract_entities(self, text: str) -> set[str]:
        """Extracts all Named Entities from a text string. Kept for callers
        that only need entities (e.g. multi-turn drift on prior turns)."""
        return self._entities_from_doc(self._parse(text))

    def _locate_entity_spans(self, doc, entity_strings: set[str]) -> list:
        """
        Finds every occurrence of each known entity string inside `doc`,
        by exact text match rather than relying on that doc's own NER
        pass. This matters because spaCy's small NER model can tag an
        entity in one text (e.g. the retrieved context) and miss the
        identical string in the other (e.g. a short generated answer) —
        without this, relation comparison would silently skip pairs it
        should be checking. `entity_strings` is the union of entities
        already confirmed by NER in *either* text, so this never invents
        a new entity — it only relocates ones we already trust are real.
        """
        if doc is None or not entity_strings:
            return []
        from spacy.matcher import PhraseMatcher
        from spacy.util import filter_spans

        matcher = PhraseMatcher(self._nlp.vocab, attr="LOWER")
        matcher.add("ENT", [self._nlp.make_doc(e) for e in entity_strings])
        spans = [doc[start:end] for _, start, end in matcher(doc)]
        return filter_spans(spans)

    @staticmethod
    def _extract_relations(doc, ent_spans: list) -> set[tuple[str, str, str]]:
        """
        (entity, relation_label, entity) triples anchored on dependency
        roles — only entities that are the grammatical SUBJECT and
        OBJECT/predicate-complement of the *same* verb/copula are paired,
        not just any two entities that happen to share a sentence. This
        reuses the dependency parse spaCy already computed for NER, so it
        costs no extra model call. Copular sentences ("X is the CEO of Y")
        use the predicate noun as the label so "is the CEO of" and
        "is CEO of" both normalize to "ceo", while "founded"/"was founded
        by" both normalize to "found". Intentionally approximate — a free
        supplementary signal, not a replacement for RAGAS's LLM-based
        claim verification.
        """
        if doc is None or len(ent_spans) < 2:
            return set()

        def entity_for_token(tok, spans):
            for sp in spans:
                if sp.start <= tok.i < sp.end:
                    return sp.text.lower().strip()
            return None

        relations: set[tuple[str, str, str]] = set()
        for sent in doc.sents:
            sent_spans = [sp for sp in ent_spans if sp.start >= sent.start and sp.end <= sent.end]
            if len(sent_spans) < 2:
                continue
            for tok in sent:
                if tok.pos_ not in ("VERB", "AUX"):
                    continue
                subs = [c for c in tok.children if c.dep_ in ("nsubj", "nsubjpass")]
                objs = [c for c in tok.children if c.dep_ in ("dobj", "attr", "acomp", "oprd")]
                for c in tok.children:
                    if c.dep_ in ("prep", "agent"):
                        objs.extend(gc for gc in c.children if gc.dep_ == "pobj")

                label = tok.lemma_.lower()
                if label == "be":
                    attr = next((c for c in tok.children if c.dep_ in ("attr", "acomp")), None)
                    if attr is not None:
                        attr_ent = entity_for_token(attr, sent_spans)
                        if attr_ent is not None:
                            # "X's TITLE is ENTITY" (e.g. "Tesla's CEO is Elon Musk"):
                            # attr IS the object entity directly. If the subject is a
                            # non-entity title noun ("CEO") with a possessive entity
                            # ("Tesla's"), re-anchor the subject to that possessive so
                            # we don't miss the real subject-object pair.
                            objs = [attr]
                            for s in list(subs):
                                if entity_for_token(s, sent_spans) is not None:
                                    continue
                                poss = next((c for c in s.children if c.dep_ == "poss"), None)
                                if poss is not None and entity_for_token(poss, sent_spans) is not None:
                                    label = s.lemma_.lower()
                                    subs = [poss]
                        else:
                            # "X is the TITLE of Y" (e.g. "John Smith is the CEO of
                            # Acme Corp"): TITLE is the relation label, Y is the object.
                            label = attr.lemma_.lower()
                            of_objs = [
                                gc for c in attr.children if c.dep_ == "prep"
                                for gc in c.children if gc.dep_ == "pobj"
                            ]
                            if of_objs:
                                objs = of_objs

                if not subs or not objs:
                    continue
                for s in subs:
                    s_ent = entity_for_token(s, sent_spans)
                    if s_ent is None:
                        continue
                    for o in objs:
                        o_ent = entity_for_token(o, sent_spans)
                        if o_ent is None or o_ent == s_ent:
                            continue
                        pair = tuple(sorted((s_ent, o_ent)))
                        relations.add((pair[0], pair[1], label))
        return relations

    @staticmethod
    def _compare_relations(
        ctx_relations:  set[tuple[str, str, str]],
        resp_relations: set[tuple[str, str, str]],
    ) -> list[dict]:
        """
        Same entity pair present in both context and answer, but with a
        different relation label => the relationship drifted even though
        no entity was "hallucinated". Catches cases like
        CEO_OF -> FOUNDED that pure entity-set overlap misses.
        """
        ctx_by_pair: dict[tuple[str, str], set[str]] = {}
        for a, b, label in ctx_relations:
            ctx_by_pair.setdefault((a, b), set()).add(label)

        drift_pairs = []
        for a, b, label in resp_relations:
            ctx_labels = ctx_by_pair.get((a, b))
            if ctx_labels and label not in ctx_labels:
                drift_pairs.append({
                    "entities":         [a, b],
                    "answer_relation":  label,
                    "context_relation": sorted(ctx_labels)[0],
                })
        return drift_pairs

    def score(
        self,
        retrieved_context: list[str] | str,
        rag_answer:        str,
        previous_response: str | None = None,  # For multi-turn drift tracking
    ) -> dict[str, Any]:
        """
        Main scoring method. Runs synchronously (no async needed — pure local).

        Args:
            retrieved_context: The source chunks the LLM was given.
            rag_answer:        The LLM's generated answer.
            previous_response: Last turn's answer (for multi-turn drift).

        Returns:
            Structured dict ready to merge into ControlPlaneState.
        """
        self._initialize()
        t0 = time.perf_counter()

        # ── Flatten context chunks into one string ─────────────
        if isinstance(retrieved_context, list):
            context_str = " ".join(retrieved_context)
        else:
            context_str = retrieved_context

        # ── Extract entities (single parse per text, reused for relations) ─
        ctx_doc   = self._parse(context_str)
        resp_doc  = self._parse(rag_answer)
        ctx_ents  = self._entities_from_doc(ctx_doc)
        resp_ents = self._entities_from_doc(resp_doc)

        # ── Hallucinated = entities in answer BUT NOT in context ─
        hallucinated_ents = resp_ents - ctx_ents

        # ── Drift score = fraction of response entities that are new ─
        total_resp_ents = len(resp_ents) or 1  # avoid division by zero
        drift_score = len(hallucinated_ents) / total_resp_ents

        # ── Overlap ratio = fraction of response entities found in context ─
        overlap = resp_ents & ctx_ents
        entity_overlap_ratio = len(overlap) / total_resp_ents

        # ── Relation drift: same entity pair, different relation ──
        # Union so an entity NER caught only on one side (common on short
        # generated sentences) is still located on both sides for comparison.
        union_ents     = ctx_ents | resp_ents
        ctx_ent_spans  = self._locate_entity_spans(ctx_doc, union_ents)
        resp_ent_spans = self._locate_entity_spans(resp_doc, union_ents)
        ctx_relations  = self._extract_relations(ctx_doc, ctx_ent_spans)
        resp_relations = self._extract_relations(resp_doc, resp_ent_spans)
        relation_drift_pairs = self._compare_relations(ctx_relations, resp_relations)
        total_resp_relations = len(resp_relations) or 1
        relation_drift_score = len(relation_drift_pairs) / total_resp_relations

        # ── Multi-turn drift (how much did entities change vs last turn?) ─
        multi_turn_drift_score = 0.0
        if previous_response:
            prev_ents           = self._extract_entities(previous_response)
            turn_new_ents       = resp_ents - prev_ents
            multi_turn_drift_score = len(turn_new_ents) / total_resp_ents

        # ── Verdict ────────────────────────────────────────────
        verdict, reasoning = self._derive_verdict(
            drift_score, entity_overlap_ratio, hallucinated_ents, relation_drift_pairs
        )

        latency_ms = (time.perf_counter() - t0) * 1000

        logger.info(
            f"[EntityDriftAgent] drift={drift_score:.3f} "
            f"overlap={entity_overlap_ratio:.3f} "
            f"hallucinated_entities={list(hallucinated_ents)} "
            f"relation_drift_pairs={relation_drift_pairs} "
            f"verdict={verdict} ({latency_ms:.0f}ms)"
        )

        return {
            "entity_drift_results": {
                "context_entities":       sorted(ctx_ents),
                "response_entities":      sorted(resp_ents),
                "hallucinated_entities":  sorted(hallucinated_ents),
                "drift_score":            round(drift_score, 4),
                "entity_overlap_ratio":   round(entity_overlap_ratio, 4),
                "multi_turn_drift_score": round(multi_turn_drift_score, 4),
                "relation_drift_pairs":   relation_drift_pairs,
                "relation_drift_score":   round(relation_drift_score, 4),
            },
            "entity_drift_verdict":    verdict,
            "entity_drift_reasoning":  reasoning,
            "entity_drift_latency_ms": round(latency_ms, 1),
        }

    @staticmethod
    def _derive_verdict(
        drift_score:           float,
        entity_overlap_ratio:  float,
        hallucinated_ents:     set[str],
        relation_drift_pairs:  list[dict],
    ) -> tuple[str, str]:
        """Priority-ordered verdict logic."""
        if drift_score >= DRIFT_SCORE_FAIL_THRESHOLD:
            return "fail", (
                f"CRITICAL: {len(hallucinated_ents)} hallucinated entit"
                f"{'y' if len(hallucinated_ents)==1 else 'ies'} detected "
                f"(drift={drift_score:.0%}): {sorted(hallucinated_ents)}. "
                "These names/numbers/facts were never in the source document."
            )
        if drift_score >= DRIFT_SCORE_WARN_THRESHOLD:
            return "warn", (
                f"WARNING: {len(hallucinated_ents)} potential hallucinated entities "
                f"(drift={drift_score:.0%}): {sorted(hallucinated_ents)}. "
                "Recommend human review."
            )
        if len(relation_drift_pairs) >= RELATION_DRIFT_FAIL_COUNT:
            return "fail", (
                f"CRITICAL: {len(relation_drift_pairs)} entity relationships changed "
                f"even though the entities themselves are in the source: "
                f"{relation_drift_pairs}. The answer misstates how these entities relate."
            )
        if entity_overlap_ratio < OVERLAP_FAIL_THRESHOLD and len(hallucinated_ents) > 0:
            return "warn", (
                f"Low entity overlap ({entity_overlap_ratio:.0%}) with "
                f"{len(hallucinated_ents)} new entities. Borderline case."
            )
        if len(relation_drift_pairs) >= RELATION_DRIFT_WARN_COUNT:
            return "warn", (
                f"Entities are grounded in the source, but the relationship between "
                f"them may have changed: {relation_drift_pairs}. Recommend human review."
            )
        return "pass", (
            f"Entity overlap={entity_overlap_ratio:.0%}  drift={drift_score:.0%}  "
            "All key entities and their relationships are consistent with the "
            "source context. ✓"
        )


# ── LangGraph-compatible node (synchronous — no async needed) ──
_agent = EntityDriftAgent()

def entity_drift_node(state: dict) -> dict:
    """
    Synchronous LangGraph node for the Performance Branch.
    Reads:  state['retrieved_context'], state['rag_answer'], state['conversation_history']
    Writes: state['entity_drift_results'], state['entity_drift_verdict'],
            state['entity_drift_reasoning']
    """
    # Extract the previous turn's answer from conversation history for drift tracking
    history = state.get("conversation_history", [])
    previous_response = None
    if history and isinstance(history, list) and len(history) >= 2:
        # Conversation history entries should have 'content' key
        last_ai_message = [
            m for m in history if m.get("role") == "assistant"
        ]
        if last_ai_message:
            previous_response = last_ai_message[-1].get("content")

    result = _agent.score(
        retrieved_context=state.get("retrieved_context", []),
        rag_answer=state.get("rag_answer", ""),
        previous_response=previous_response,
    )
    return {**state, **result}


# ── CLI smoke test ─────────────────────────────────────────────
if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)

    agent = EntityDriftAgent()

    print("\n" + "="*60)
    print("TEST 1: Faithful answer (expected: PASS)")
    print("="*60)
    r = agent.score(
        retrieved_context=[
            "The CEO of Acme Corp is John Smith. "
            "The company was founded in 1995 in New York."
        ],
        rag_answer="John Smith is the CEO of Acme Corp, founded in New York in 1995.",
    )
    print(json.dumps(r, indent=2))

    print("\n" + "="*60)
    print("TEST 2: Hallucinated entity (expected: FAIL)")
    print("="*60)
    r = agent.score(
        retrieved_context=[
            "The CEO of Acme Corp is John Smith. "
            "The company was founded in 1995 in New York."
        ],
        rag_answer=(
            "Jane Doe is the CEO of Acme Corp. "
            "The company was founded in 1990 in San Francisco by Robert Lee."
        ),
    )
    print(json.dumps(r, indent=2))

    print("\n" + "="*60)
    print("TEST 3: Relation drift, entities all grounded (expected: WARN/FAIL)")
    print("="*60)
    r = agent.score(
        retrieved_context=["Tesla's CEO is Elon Musk."],
        rag_answer="Tesla was founded by Elon Musk.",
    )
    print(json.dumps(r, indent=2))
