"""
LangGraph state schema for the ControlPlane.ai workflow.

One TypedDict carried through every node. Fields are grouped by pipeline stage.
`total=False` so nodes only need to return the keys they actually write.

Reducer note: `node_timings` and `llm_calls` use an additive reducer so parallel
branches (performance / responsibility) can both append without clobbering.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict


def _merge_dicts(left: Optional[dict], right: Optional[dict]) -> dict:
    out = dict(left or {})
    out.update(right or {})
    return out


def _last(left: Any, right: Any) -> Any:
    """Last-write-wins reducer - lets parallel branches both touch a scalar field."""
    return right if right is not None else left


class ControlPlaneState(TypedDict, total=False):
    # ---- Core query --------------------------------------------------------------
    original_query: str            # immutable, exactly what the user typed
    guarded_query: str             # after PII masking (what everything downstream uses)
    updated_query: str             # after a hallucination-retry rewrite or HITL enrichment
    conversation_id: str

    # ---- Guardrails --------------------------------------------------------------
    guardrail_flags: List[str]     # e.g. ["pii:EMAIL", "pii:PHONE"]
    guardrail_pii_spans: List[Dict[str, str]]
    blocked: bool
    block_reason: Optional[str]
    block_category: Optional[str]  # "prompt_injection" | "jailbreak"

    # ---- Semantic cache --------------------------------------------------------------
    cache_hit: bool
    cache_similarity: Optional[float]
    cached_answer: Optional[str]
    cached_meta: Optional[Dict[str, Any]]

    # ---- RAG router (the one main agent) -----------------------------------------
    selected_kb: Optional[str]     # one of KB_IDS or "none"
    router_reason: Optional[str]
    router_confidence: Optional[float]

    # ---- Retrieval --------------------------------------------------------------
    vector_chunks: List[Dict[str, Any]]
    bm25_chunks: List[Dict[str, Any]]
    rrf_chunks: List[Dict[str, Any]]        # final top-k fused context
    retrieval_meta: Dict[str, Any]          # per-branch latency, counts

    # ---- Answer generation --------------------------------------------------------------
    answer: str
    model_used: Optional[str]
    model_category: Optional[str]           # light | medium | heavy | ...
    model_tier: Optional[int]               # 1|2|3  (feeds XGBoost, matches feature_engineering)
    answer_temperature: float
    token_stats: Dict[str, int]             # {prompt_tokens, completion_tokens, total_tokens}
    cost_usd: float                         # Gemini > 0, Groq forced 0

    # ---- Performance branch --------------------------------------------------------------
    ragas_scores: Dict[str, float]          # faithfulness, answer_relevancy, context_coverage
    ragas_verdict: Optional[str]
    xgboost_prob: Optional[float]           # probability of hallucination
    xgboost_risk: Optional[str]             # LOW|MODERATE|HIGH|CRITICAL
    xgboost_features: Dict[str, float]
    entity_drift: Dict[str, Any]            # drift_score, hallucinated_entities, verdict, ...
    perf_verdict: Optional[str]             # "pass" | "hallucinated" | "need_human"
    perf_reasoning: Optional[str]
    perf_suggestion: Optional[str]          # retrieval-focused query rewrite (when hallucinated)
    perf_score: Optional[float]
    detector_votes: Dict[str, str]
    ragas_unsupported: List[str]
    perf_branch_ms: Optional[float]

    # ---- Responsibility branch --------------------------------------------------------------
    resp_vector_chunks: List[Dict[str, Any]]
    resp_bm25_chunks: List[Dict[str, Any]]
    resp_graph_chunks: List[Dict[str, Any]]
    resp_rrf_chunks: List[Dict[str, Any]]
    resp_retrieval_meta: Dict[str, Any]
    toxicity: Dict[str, Dict[str, Any]]    # {detoxify:{prob,label}, unitary:{...}, snlp:{...}}
    toxicity_max: Optional[float]
    resp_status: Optional[str]             # "safe" | "unsafe" | "uncertain"
    resp_reasoning: Optional[str]
    violated_rules: List[str]
    resp_report: Optional[str]             # structured LLM report (only when flagged)
    evidence_chunks: List[Dict[str, Any]]
    resp_eval_ms: Optional[float]
    resp_branch_ms: Optional[float]
    _resp_llm_call: Optional[Dict[str, Any]]

    # ---- Retry / HITL --------------------------------------------------------------
    retry_count: int
    hitl_count: int
    hitl_needed: bool
    hitl_question: Optional[str]
    hitl_context: Optional[Dict[str, Any]]
    hitl_response: Optional[str]

    # ---- Final --------------------------------------------------------------
    _next: Optional[str]                    # internal routing hint out of aggregate
    final_decision: Optional[str]          # allow | block | cache | harmful | hitl
    final_answer: str
    final_verdict_badges: List[str]

    # ---- Observability --------------------------------------------------------------
    stage: Annotated[str, _last]            # current pipeline stage (drives Tab 3)
    stages_visited: Annotated[List[str], operator.add]
    node_timings: Annotated[Dict[str, float], _merge_dicts]
    llm_calls: Annotated[List[Dict[str, Any]], operator.add]
    langsmith_run_id: Optional[str]
    started_at: float
    total_latency_ms: Optional[float]


# Stage constants (kept in one place so Tab 3 and the nodes never drift).
class Stage:
    START = "start"
    GUARDRAILS = "guardrails"
    CACHE = "semantic_cache"
    ROUTER = "rag_router"
    RETRIEVAL = "retrieval"
    ANSWER = "answer_generation"
    PERFORMANCE = "performance"
    RESPONSIBILITY = "responsibility"
    AGGREGATE = "aggregate"
    HALLUCINATION_RETRY = "hallucination_retry"
    HITL = "human_in_the_loop"
    FINALIZE = "finalize"
    DONE = "done"

    ORDER = [
        START, GUARDRAILS, CACHE, ROUTER, RETRIEVAL, ANSWER,
        PERFORMANCE, RESPONSIBILITY, AGGREGATE, HITL, FINALIZE, DONE,
    ]


def new_state(query: str, conversation_id: str = "default") -> ControlPlaneState:
    """Build a fully-initialised state for a fresh query."""
    import time

    return ControlPlaneState(
        original_query=query,
        guarded_query=query,
        updated_query=query,
        conversation_id=conversation_id,
        guardrail_flags=[],
        blocked=False,
        block_reason=None,
        block_category=None,
        cache_hit=False,
        cache_similarity=None,
        cached_answer=None,
        cached_meta=None,
        selected_kb=None,
        router_reason=None,
        router_confidence=None,
        vector_chunks=[],
        bm25_chunks=[],
        rrf_chunks=[],
        retrieval_meta={},
        answer="",
        model_used=None,
        model_category=None,
        model_tier=None,
        answer_temperature=0.2,
        token_stats={},
        cost_usd=0.0,
        ragas_scores={},
        ragas_verdict=None,
        xgboost_prob=None,
        xgboost_risk=None,
        xgboost_features={},
        entity_drift={},
        perf_verdict=None,
        perf_reasoning=None,
        perf_suggestion=None,
        perf_score=None,
        resp_vector_chunks=[],
        resp_bm25_chunks=[],
        resp_graph_chunks=[],
        resp_rrf_chunks=[],
        resp_retrieval_meta={},
        toxicity={},
        toxicity_max=None,
        resp_status=None,
        resp_reasoning=None,
        violated_rules=[],
        resp_report=None,
        evidence_chunks=[],
        retry_count=0,
        hitl_count=0,
        hitl_needed=False,
        hitl_question=None,
        hitl_context=None,
        hitl_response=None,
        final_decision=None,
        final_answer="",
        final_verdict_badges=[],
        stage=Stage.START,
        stages_visited=[Stage.START],
        node_timings={},
        llm_calls=[],
        langsmith_run_id=None,
        started_at=time.time(),
        total_latency_ms=None,
    )
