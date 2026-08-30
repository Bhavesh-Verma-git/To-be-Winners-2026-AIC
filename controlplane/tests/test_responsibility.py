from controlplane.responsibility.evaluator import evaluate_responsibility


def _state(tox_max, chunks=None, answer="some answer"):
    return dict(
        answer=answer,
        guarded_query="q",
        toxicity={"detoxify": {"prob": tox_max, "label": "toxic" if tox_max >= 0.5 else "non-toxic"},
                  "unitary": {"prob": tox_max * 0.9, "label": "n"},
                  "snlp": {"prob": tox_max * 0.8, "label": "n"}},
        toxicity_max=tox_max,
        resp_rrf_chunks=chunks or [],
    )


def test_low_toxicity_is_safe():
    out = evaluate_responsibility(_state(0.02))
    assert out["resp_status"] == "safe"
    assert out["resp_report"] is None  # no LLM call on the safe path


def test_high_toxicity_is_unsafe():
    out = evaluate_responsibility(_state(0.95))
    assert out["resp_status"] in {"unsafe", "uncertain"}  # mock LLM may clear 'uncertain'
    # hard toxicity keeps it unsafe even if the mock LLM says compliant
    assert out["resp_status"] == "unsafe"


def test_risky_answer_plus_prohibiting_clause_flags():
    chunks = [{"text": "Article 5 prohibits AI systems that use subliminal techniques to manipulate persons.",
               "title": "EU AI Act Article 5", "score": 0.05,
               "metadata": {"law_or_article": "EU AI Act Article 5"}}] * 2
    out = evaluate_responsibility(
        _state(0.05, chunks, answer="Deploy subliminal micro-cues to manipulate users into purchasing subscriptions.")
    )
    assert out["resp_status"] in {"unsafe", "uncertain"}
    assert "clause_signal" in out["resp_reasoning"]


def test_benign_answer_with_legal_chunks_stays_safe():
    chunks = [{"text": "This article prohibits discriminatory practices and unlawful profiling.",
               "title": "EU AI Act", "score": 0.05, "metadata": {"law_or_article": "EU AI Act"}}] * 3
    out = evaluate_responsibility(_state(0.02, chunks, answer="You can return an item within 30 days with a receipt."))
    assert out["resp_status"] == "safe"
    assert out["resp_report"] is None
