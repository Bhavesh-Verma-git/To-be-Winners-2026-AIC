"""
============================================================
  ControlPlane.ai - LangGraph Router Graph
  WITH PERFORMANCE BRANCH (Hallucination Detection)

  Architecture:
      [START]
         |
    [router_node]       <- Qwen: classifies query to 1 of 5 routes
         |
    [agent_node]        <- RAG agent generates answer
         |
    [perf_fan_out]      <- triggers 3 parallel evaluators
         |
    [xgboost] [ragas] [entity_drift]   <- parallel
         |
    [perf_fan_in]       <- majority-vote referee -> pass/retry/hitl
         |
       [END]
============================================================
"""

import os
import time
from typing import TypedDict, Optional
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq

from master_router.contract import RAGAgentOutput, VALID_ROUTES
from master_router.wrappers import dispatch

# Performance Branch imports
from master_router.performance_branch.ragas_agent import RAGASAgent
from master_router.performance_branch.entity_drift_agent import EntityDriftAgent
from master_router.performance_branch.performance_evaluator import PerformanceEvaluator, PerformanceSignals

# Singletons - loaded once, reused every query
_ragas_agent = RAGASAgent()
_entity_drift_agent = EntityDriftAgent()
_perf_evaluator = PerformanceEvaluator()


# ----------------------------------------------------------------
# Graph State
# ----------------------------------------------------------------
class RouterState(TypedDict):
    # Core routing
    user_query:   str
    route:        Optional[str]
    agent_output: Optional[RAGAgentOutput]
    trace:        list
    router_ms:    Optional[float]
    agent_ms:     Optional[float]
    # Performance Branch inputs
    rag_answer:        Optional[str]
    retrieved_context: Optional[list]
    model_name:        Optional[str]
    temperature:       Optional[float]
    # XGBoost
    xgboost_scores: Optional[dict]
    # RAGAS
    ragas_scores:     Optional[dict]
    ragas_verdict:    Optional[str]
    ragas_reasoning:  Optional[str]
    ragas_latency_ms: Optional[float]
    # Entity Drift
    entity_drift_results:    Optional[dict]
    entity_drift_verdict:    Optional[str]
    entity_drift_reasoning:  Optional[str]
    entity_drift_latency_ms: Optional[float]
    # Final decision
    performance_evaluator_decision:  Optional[str]
    performance_evaluator_reasoning: Optional[str]
    performance_score: Optional[float]
    detector_votes:    Optional[dict]
    perf_ms:           Optional[float]


# ----------------------------------------------------------------
# Node 1: Router
# ----------------------------------------------------------------
ROUTER_PROMPT = """You are a query router for an enterprise AI system.
Your ONLY job is to read the user question and output a single JSON object
with the key "route" set to exactly one of these 5 values:

  "customer_support"  -> product complaints, orders, refunds, billing, account
  "hr_policy"         -> HR rules, leave policy, salary, attendance, holidays
  "azure_docs"        -> Microsoft Azure App Service: deployment, scaling, config
  "toxicity"          -> content moderation, harmful or offensive content
  "decision_support"  -> data-driven recommendations, strategic decisions, risk
  "unknown"           -> does not match any category

Examples:
Q: "How do I get a refund?"              -> {{"route": "customer_support"}}
Q: "How many sick leaves do I get?"      -> {{"route": "hr_policy"}}
Q: "az webapp config appsettings set"   -> {{"route": "azure_docs"}}
Q: "Is this message offensive?"         -> {{"route": "toxicity"}}
Q: "What is the best market strategy?"  -> {{"route": "decision_support"}}
Q: "What is the capital of France?"     -> {{"route": "unknown"}}

Respond ONLY with the JSON object. Question: {query}"""


def router_node(state: RouterState) -> RouterState:
    query = state["user_query"]
    trace = state.get("trace", [])
    trace.append({"step": "Router", "status": "running",
                  "detail": f"Classifying: '{query[:60]}...'"})

    llm = ChatGroq(model_name="qwen/qwen3.6-27b", temperature=0,
                   max_tokens=1024, api_key=GROQ_API_KEY)

    t0  = time.time()
    raw = llm.invoke(ROUTER_PROMPT.format(query=query)).content
    if "</think>" in raw:
        raw = raw.split("</think>")[-1].strip()

    route     = "unknown"
    raw_lower = raw.lower()
    for valid_r in VALID_ROUTES.keys():
        if valid_r != "unknown" and (
            f'"{valid_r}"' in raw_lower or f"'{valid_r}'" in raw_lower
        ):
            route = valid_r
            break
    if route == "unknown":
        for valid_r in VALID_ROUTES.keys():
            if valid_r != "unknown" and valid_r in raw_lower:
                route = valid_r
                break

    router_ms = (time.time() - t0) * 1000
    trace[-1]["status"] = "done"
    trace[-1]["detail"] = f"Route: **{VALID_ROUTES.get(route, route)}** ({router_ms:.0f}ms)"
    return {**state, "route": route, "trace": trace, "router_ms": router_ms}


# ----------------------------------------------------------------
# Node 2: Agent Dispatcher
# ----------------------------------------------------------------
def agent_node(state: RouterState) -> RouterState:
    query       = state["user_query"]
    route       = state.get("route", "unknown")
    trace       = state.get("trace", [])
    agent_label = VALID_ROUTES.get(route, "Unknown")

    trace.append({"step": agent_label, "status": "running",
                  "detail": "Searching knowledge base..."})

    t0       = time.time()
    output   = dispatch(route, query)
    agent_ms = (time.time() - t0) * 1000

    trace[-1]["status"] = "done"
    detail = f"Retrieved {output['retrieved_n']} chunks | {agent_ms:.0f}ms"
    if output.get("source_url"):
        detail += f" | {output['source_url'][:40]}..."
    if output.get("error"):
        trace[-1]["status"] = "error"
        detail = f"Error: {output['error'][:80]}"
    trace[-1]["detail"] = detail
    return {**state, "agent_output": output, "trace": trace, "agent_ms": agent_ms}


# ----------------------------------------------------------------
# Node 3: Performance Fan-Out
# ----------------------------------------------------------------
def perf_fan_out_node(state: RouterState) -> RouterState:
    trace = state.get("trace", [])
    trace.append({"step": "Performance Branch", "status": "running",
                  "detail": "Running 3 hallucination detectors in parallel..."})

    output     = state.get("agent_output") or {}
    rag_answer = output.get("rag_answer", "")
    source     = output.get("source", "")
    retrieved_context = [source] if source else [rag_answer[:500]]

    return {
        **state,
        "rag_answer": rag_answer,
        "retrieved_context": retrieved_context,
        "model_name": "qwen/qwen3.6-27b",
        "temperature": 0.3,
        "trace": trace,
        "xgboost_scores": None,
        "ragas_scores": None, "ragas_verdict": None,
        "ragas_reasoning": None, "ragas_latency_ms": None,
        "entity_drift_results": None, "entity_drift_verdict": None,
        "entity_drift_reasoning": None, "entity_drift_latency_ms": None,
        "performance_evaluator_decision": None,
        "performance_evaluator_reasoning": None,
        "performance_score": None, "detector_votes": None, "perf_ms": None,
    }


# ----------------------------------------------------------------
# Node 4a: XGBoost
# ----------------------------------------------------------------
def xgboost_node(state: RouterState) -> RouterState:
    t0 = time.time()
    try:
        from master_router.performance_branch.xgboost_agent import XGBoostHallucinationAgent
        agent  = XGBoostHallucinationAgent()
        result = agent.score(
            context=state.get("retrieved_context", [""])[0],
            response=state.get("rag_answer", ""),
            model_name=state.get("model_name", "unknown"),
            temperature=state.get("temperature", 0.3),
        )
        xgboost_scores = {
            "hallucination_prob": result["hallucination_probability"],
            "is_hallucination":   result["is_hallucination"],
            "risk_level":         result["risk_level"],
            "latency_ms":         round((time.time() - t0) * 1000, 1),
        }
    except Exception as exc:
        xgboost_scores = {
            "hallucination_prob": 0.3, "is_hallucination": False,
            "risk_level": "LOW", "error": str(exc),
            "latency_ms": round((time.time() - t0) * 1000, 1),
        }
    return {"xgboost_scores": xgboost_scores}


# ----------------------------------------------------------------
# Node 4b: RAGAS
# ----------------------------------------------------------------
def ragas_eval_node(state: RouterState) -> dict:
    result = _ragas_agent.score_sync(
        query=state.get("user_query", ""),
        rag_answer=state.get("rag_answer", ""),
        retrieved_context=state.get("retrieved_context", []),
    )
    return result


# ----------------------------------------------------------------
# Node 4c: Entity Drift
# ----------------------------------------------------------------
def entity_drift_eval_node(state: RouterState) -> dict:
    result = _entity_drift_agent.score(
        retrieved_context=state.get("retrieved_context", []),
        rag_answer=state.get("rag_answer", ""),
    )
    return result


# ----------------------------------------------------------------
# Node 5: Performance Fan-In (Majority Vote Referee)
# ----------------------------------------------------------------
def perf_fan_in_node(state: RouterState) -> RouterState:
    t0      = time.time()
    signals = PerformanceSignals.from_state(state)
    result  = _perf_evaluator.aggregate(signals)
    perf_ms = (time.time() - t0) * 1000

    trace    = state.get("trace", [])
    decision = result["performance_evaluator_decision"]
    score    = result["performance_score"]
    status_map = {"pass": "done", "retry": "error", "hitl": "warning"}
    if trace:
        trace[-1]["status"] = status_map.get(decision, "done")
        trace[-1]["detail"] = (
            f"Decision: **{decision.upper()}** | "
            f"Score={score:.2f} | Votes={result['detector_votes']}"
        )
    return {**state, **result, "trace": trace, "perf_ms": round(perf_ms, 1)}


# ----------------------------------------------------------------
# Conditional Edge + Build Graph
# ----------------------------------------------------------------
def route_decision(state: RouterState) -> str:
    return "agent"


def build_graph():
    builder = StateGraph(RouterState)
    builder.add_node("router",       router_node)
    builder.add_node("agent",        agent_node)
    builder.add_node("perf_fan_out", perf_fan_out_node)
    builder.add_node("xgboost",      xgboost_node)
    builder.add_node("ragas",        ragas_eval_node)
    builder.add_node("entity_drift", entity_drift_eval_node)
    builder.add_node("perf_fan_in",  perf_fan_in_node)

    builder.set_entry_point("router")
    builder.add_conditional_edges("router", route_decision, {"agent": "agent"})
    builder.add_edge("agent",        "perf_fan_out")
    builder.add_edge("perf_fan_out", "xgboost")
    builder.add_edge("perf_fan_out", "ragas")
    builder.add_edge("perf_fan_out", "entity_drift")
    builder.add_edge("xgboost",      "perf_fan_in")
    builder.add_edge("ragas",        "perf_fan_in")
    builder.add_edge("entity_drift", "perf_fan_in")
    builder.add_edge("perf_fan_in",  END)
    return builder.compile()


def run(query: str) -> RouterState:
    graph = build_graph()
    return graph.invoke({
        "user_query": query, "route": None, "agent_output": None,
        "trace": [], "router_ms": None, "agent_ms": None,
        "rag_answer": None, "retrieved_context": None,
        "model_name": None, "temperature": None,
        "xgboost_scores": None,
        "ragas_scores": None, "ragas_verdict": None,
        "ragas_reasoning": None, "ragas_latency_ms": None,
        "entity_drift_results": None, "entity_drift_verdict": None,
        "entity_drift_reasoning": None, "entity_drift_latency_ms": None,
        "performance_evaluator_decision": None,
        "performance_evaluator_reasoning": None,
        "performance_score": None, "detector_votes": None, "perf_ms": None,
    })


if __name__ == "__main__":
    q = "How many sick leaves do I get per year?"
    print(f"Q: {q}")
    r = run(q)
    print(f"Route : {r['route']}")
    print(f"Answer: {r['agent_output']['rag_answer'][:200]}...")
    print(f"XGBoost   : {r.get('xgboost_scores',{}).get('risk_level')} prob={r.get('xgboost_scores',{}).get('hallucination_prob',0):.3f}")
    print(f"RAGAS     : {r.get('ragas_verdict')} faith={r.get('ragas_scores',{}).get('faithfulness',0):.3f}")
    print(f"Entity    : {r.get('entity_drift_verdict')} drift={r.get('entity_drift_results',{}).get('drift_score',0):.3f}")
    print(f"DECISION  : {str(r.get('performance_evaluator_decision','N/A')).upper()}")
    print(f"Score     : {r.get('performance_score',0):.3f}")
    print(f"Timings   : Router={r['router_ms']:.0f}ms Agent={r['agent_ms']:.0f}ms Perf={r.get('perf_ms',0):.0f}ms")
