from controlplane.performance.evaluator import evaluate_performance


def _state(**kw):
    base = dict(
        updated_query="q",
        answer="a",
        rrf_chunks=[{"text": "c"}],
        ragas_scores={"faithfulness": 0.9, "answer_relevancy": 0.9, "context_coverage": 0.9},
        ragas_verdict="grounded",
        xgboost_prob=0.1,
        entity_drift={"entity_drift_verdict": "pass", "entity_drift_results": {"drift_score": 0.0}},
        retry_count=0,
    )
    base.update(kw)
    return base


def test_all_clean_passes():
    assert evaluate_performance(_state())["perf_verdict"] == "pass"


def test_fabricated_facts_is_hallucinated_with_suggestion():
    out = evaluate_performance(
        _state(
            entity_drift={
                "entity_drift_verdict": "fail",
                "entity_drift_results": {
                    "drift_score": 0.7,
                    "hallucinated_entities": ["jane doe", "san francisco", "1990", "robert lee"],
                    "relation_drift_pairs": [],
                },
            }
        )
    )
    assert out["perf_verdict"] == "hallucinated"
    assert out.get("perf_suggestion")


def test_xgb_confident_plus_ragas_contradiction_is_hallucinated():
    out = evaluate_performance(
        _state(xgboost_prob=0.85, ragas_verdict="hallucinated", ragas_scores={"faithfulness": 0.2})
    )
    assert out["perf_verdict"] == "hallucinated"


def test_low_ragas_alone_still_passes():
    # a synthesised-but-fine answer: low literal faithfulness, but xgb + entity are clean
    out = evaluate_performance(
        _state(xgboost_prob=0.3, ragas_verdict="partially_grounded",
               ragas_scores={"faithfulness": 0.55, "answer_relevancy": 0.9, "context_coverage": 0.7})
    )
    assert out["perf_verdict"] == "pass"


def test_missing_info_answer_with_thin_retrieval_triggers_need_human():
    out = evaluate_performance(
        _state(answer="The knowledge base does not contain enough information to answer this.",
               ragas_scores={"faithfulness": 0.5, "answer_relevancy": 0.3, "context_coverage": 0.2})
    )
    assert out["perf_verdict"] == "need_human"
    assert out.get("hitl_needed") and out.get("hitl_question")


def test_missing_info_answer_with_good_retrieval_does_not_hitl():
    # retrieval was fine; if the model still refuses we deliver its answer, not HITL
    out = evaluate_performance(
        _state(answer="The knowledge base does not contain enough information to answer this.",
               ragas_scores={"faithfulness": 0.9, "answer_relevancy": 0.9, "context_coverage": 0.9})
    )
    assert out["perf_verdict"] == "pass"


def test_no_context_triggers_need_human():
    assert evaluate_performance(_state(rrf_chunks=[]))["perf_verdict"] == "need_human"
