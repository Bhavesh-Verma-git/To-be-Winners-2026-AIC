# ControlPlane.ai — Complete Production Architecture
### Accenture AI Innovation Challenge · Round 2

> **Document scope:** Technically implementable, non-vague, non-hallucinated, parallelized LangGraph + LiteLLM agentic RAG architecture. Every component from your sketched PDF and 1.md notes is preserved and converted into a concrete engineering specification. Every Accenture requirement is mapped explicitly.

---

## 0. OFFICIAL ACCENTURE REQUIREMENT MAPPING

The following table accounts for **every** requirement from the official problem statement. Nothing is omitted.

| # | Accenture Requirement | Where Implemented | Component | Expected Output |
|---|---|---|---|---|
| R1 | Real-time evaluation of AI responses (not post-hoc logging) | Inline middleware — runs inside LangGraph before final response node | `parallel_eval_fan_out` node + all eval branches | `decision: pass\|edit\|block\|escalate` within ~300ms |
| R2 | Score on Performance axis (correctness / hallucination) | `performance_branch` parallel node | RAGAS Agent + XGBoost Agent + Entity Drift Agent | `performance_score ∈ [0,1]` |
| R3 | Score on Cost axis (token/compute anomaly) | `cost_agent` node | Isolation Forest / XGBoost anomaly model | `cost_flag: normal\|anomalous`, `tc_score` |
| R4 | Score on Responsibility axis (bias/toxicity/PII) | `responsibility_branch` parallel node | Responsibility RAG Agent + Toxicity Agent | `responsibility_score ∈ [0,1]`, `violated_clauses` |
| R5 | Decision logic: pass / edit / block / escalate | `final_evaluator_agent` node | LLM-based aggregator | `decision` field in state |
| R6 | Explanation citing exact policy clause | `responsibility_rag_agent` node | Hybrid RAG over EU AI Act / NIST / ISO 42001 KG | `violated_law`, `evidence_chunks` |
| R7 | Different risk tolerance per use case (HR / Customer / Internal / Decision) | `rag_router` conditional edge + per-use-case policy configs | Router node reads `use_case` from state | Routed sub-pipeline |
| R8 | Detection techniques: rule-based, embedding/anomaly, AI-as-judge, PII/entity | Input guardrail + XGBoost + entity drift + NLI judge | `input_guardrail_node`, `xgboost_agent`, `entity_drift_agent` | flags, scores |
| R9 | Confidence scoring, tiered responses | `final_evaluator_agent` aggregated confidence score | LLM evaluator + threshold logic | tiered decision |
| R10 | Parallel checks to protect latency | Fan-out at `parallel_eval_fan_out` | `performance_branch` ∥ `responsibility_branch` | merged in `eval_fan_in` |
| R11 | Governance — configurable policy layer by use case / geography / risk appetite | `policy_config` injected into state at `rag_router` | Per-use-case YAML/JSON policy config | `active_policy` in state |
| R12 | Clear audit trail behind every decision | LangSmith trace per run + state logged to persistent store | LangSmith SDK + `langchain_store` write node | Full trace per interaction |
| R13 | Feedback loops — flagged/overridden cases improve detection | HITL queue + LangSmith `create_feedback` + XGBoost retraining pipeline | `hitl_interrupt_node` + feedback write | Retrained classifier cycle |
| R14 | Metrics & monitoring — false positive/negative rates, system trustworthiness | LangSmith dashboard + stored eval scores | `langsmith_log_node` | Dashboards, FP/FN rates |
| R15 | Multi-turn / compounding risk awareness | LangGraph state carries conversation history + retry count | `ConversationHistory` in state | Correct multi-turn context |
| R16 | Regulatory variance by geography | Policy config layer with `geography` field | `policy_config` YAML | Region-specific clause retrieval |
| R17 | Working prototype demonstrating core mechanism | All components below must be wired and runnable | Full LangGraph graph | End-to-end demo |
| R18 | Tens of thousands of interactions/week scale | Async LangGraph execution + Redis semantic cache + vLLM local models | All throughput-relevant nodes | Sub-300ms median latency |
| R19 | Mix of well-governed and loosely governed data sources | HR/Customer/Internal/Decision RAG agents with different data provenance | `rag_router` + `use_case_rag_agent` | Source-tagged retrieved chunks |

---

## 1. SYSTEM-LEVEL ARCHITECTURE (COMPLETE FLOW)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE / API                          │
│    Input: {user_query, use_case, conversation_history}               │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   INPUT GUARDRAIL NODE                               │
│  • Prompt injection detection (rule-based regex + embedding anomaly) │
│  • Jailbreak detection (LLM classifier via Groq free-tier)           │
│  • PII detection (presidio / spaCy NER)                              │
│  FAILS → Block immediately, return structured error                  │
│  PASS  → Continue                                                    │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│               SEMANTIC CACHE CHECK NODE                              │
│  • Embed query (all-MiniLM-L6-v2, local)                            │
│  • LiteLLM semantic cache (Redis + Qdrant backend)                   │
│  • Similarity threshold ≥ 0.95 → CACHE HIT                          │
│  CACHE HIT  → Return cached {rag_answer, scores} immediately        │
│  CACHE MISS → Continue to RAG                                        │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    RAG ROUTER NODE                                   │
│  • Reads use_case from state                                         │
│  • Loads matching policy_config (YAML per use case)                  │
│  • Routes to correct sub-RAG agent                                   │
│  • Routes: HR_Policy | Customer_Support | Internal_KB | Decision_Support│
└─────────────────────────┬───────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│               AGENTIC RAG AGENT (per use-case)                       │
│  Hybrid Retrieval:                                                   │
│  ├── Vector Search (Chroma/FAISS + sentence-transformers)            │
│  ├── BM25 (rank_bm25)                                                │
│  ├── Reciprocal Rank Fusion (RRF) score merging                                                                │
│  Context Trimming: cross-encoder relevance scoring → drop bottom 30% │
│  Generation: LiteLLM routed LLM call (model tier = MEDIUM)          │
│  Outputs: {user_query, retrieved_context, rag_answer}                │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │  PARALLEL EVAL FAN-OUT │  (LangGraph Send() API)
              └──────┬────────────────┘
                     │
        ┌────────────┴─────────────┐
        │                          │
        ▼                          ▼
┌───────────────┐        ┌─────────────────────┐
│  PERFORMANCE  │        │   RESPONSIBILITY     │
│    BRANCH     │        │      BRANCH          │
│               │        │                      │
│ ┌───────────┐ │        │ ┌──────────────────┐ │
│ │RAGAS Agent│ │        │ │ Resp. RAG Agent  │ │
│ └─────┬─────┘ │        │ │(EU AI Act/NIST KB)│ │
│       │  ∥    │        │ └────────┬─────────┘ │
│ ┌─────┴─────┐ │        │          │    ∥       │
│ │ XGBoost   │ │        │ ┌────────┴─────────┐ │
│ │  Agent    │ │        │ │  TOXICITY AGENT  │ │
│ └─────┬─────┘ │        │ │  (3 models ∥)    │ │
│       │  ∥    │        │ │ ①Detoxify        │ │
│ ┌─────┴─────┐ │        │ │ ②unitary/toxic-bert│ │
│ │Entity Drift│ │        │ │ ③s-nlp/roberta   │ │
│ │  Agent    │ │        │ └──────────────────┘ │
│ └───────────┘ │        └──────────┬───────────┘
└───────┬───────┘                   │
        │                           │
        ▼                           ▼
┌───────────────┐        ┌──────────────────────┐
│ LLM Perf.     │        │ Resp. LLM Evaluator  │
│ Evaluator     │        │ (aggregates toxicity +│
│               │        │  law retrieval)       │
└───────┬───────┘        └──────────┬───────────┘
        │                           │
        └──────────┬────────────────┘
                   ▼
        ┌──────────────────────┐
        │   EVAL FAN-IN NODE   │
        │   (merge both scores)│
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │  FINAL DECISION NODE │
        │  pass→Allow Answer   │
        │  unsafe→BLOCK+Explain│
        │  uncertain→HITL      │
        │  perf_fail→RETRY(≤1) │
        └──────────┬───────────┘
                   │
       ┌───────────┼──────────────┐
       │           │              │
       ▼           ▼              ▼
  ┌─────────┐ ┌─────────┐  ┌──────────────┐
  │  ALLOW  │ │  BLOCK  │  │  HITL QUEUE  │
  │ Answer  │ │Explain+ │  │ interrupt()  │
  │ to User │ │Evidence │  │ Human input  │
  └─────────┘ └─────────┘  └──────────────┘
                                 │ (resume after human)
                                 ▼
                        ┌──────────────────┐
                        │ Re-enter RAG with│
                        │ updated query    │
                        │ (max 1 HITL retry)│
                        └──────────────────┘
```

**Observability (always running, not a blocking node):**
```
ALL NODES → LangSmith trace SDK → Dashboard (cost + latency + graphs)
```

---

## 2. LANGGRAPH STATE DESIGN

```python
from typing import TypedDict, Optional, List, Dict, Any
from langgraph.graph import add_messages

class ControlPlaneState(TypedDict):
    # ── Core Query ────────────────────────────────────────────────
    original_query: str                # immutable first user query
    updated_query: str                 # modified by HITL or perf-retry
    use_case: str                      # "hr_policy"|"customer_support"|"internal_kb"|"decision_support"
    conversation_history: List[Dict]   # multi-turn messages list
    active_policy: Dict[str, Any]      # loaded from YAML for this use_case + geography
    geography: str                     # "EU"|"US"|"APAC" etc.

    # ── RAG Outputs ───────────────────────────────────────────────
    retrieved_context: List[str]       # list of chunk strings
    retrieval_metadata: List[Dict]     # {source, score, chunk_id, retrieval_type}
    rag_answer: str                    # generated answer

    # ── Cache ─────────────────────────────────────────────────────
    cache_hit: bool
    cached_response: Optional[Dict]

    # ── Input Guardrail ───────────────────────────────────────────
    input_guardrail_passed: bool
    guardrail_flags: List[str]         # ["prompt_injection","pii_detected",...]

    # ── Performance Evaluation ────────────────────────────────────
    ragas_scores: Dict[str, float]     # faithfulness, answer_relevancy, context_recall, context_precision
    xgboost_scores: Dict[str, float]   # overlap_score, entity_consistency, nli_contradiction, hallucination_prob
    entity_drift_results: Dict         # {drifted_entities: [...], drift_score: float}
    token_stats: Dict[str, int]        # {input_tokens, output_tokens, latency_ms}
    cost_flag: str                     # "normal"|"anomalous"
    tc_score: float                    # Total Cost from formula
    performance_evaluator_decision: str  # "pass"|"retry"|"hitl"
    performance_evaluator_reasoning: str

    # ── Responsibility RAG ────────────────────────────────────────
    resp_retrieved_laws: List[str]     # exact clause text retrieved
    resp_retrieval_scores: List[float] # semantic similarity scores
    resp_kg_path: Optional[str]        # KG traversal result: Policy→RiskCategory→Remediation
    responsibility_score: float        # aggregated responsibility risk score ∈ [0,1]
    violated_law: Optional[str]        # specific violated clause
    evidence_chunks: List[str]

    # ── Toxicity Agent ────────────────────────────────────────────
    detoxify_scores: Dict[str, float]  # {toxicity, severe_toxicity, obscene, identity_attack, insult, threat, sexual_explicit}
    detoxify_label: str
    toxic_bert_score: float
    toxic_bert_label: str
    roberta_tox_score: float
    roberta_tox_label: str
    toxicity_aggregate: float          # max or weighted ensemble of 3 models
    toxicity_flagged: bool

    # ── Responsibility Evaluator ──────────────────────────────────
    responsibility_evaluator_decision: str   # "safe"|"unsafe"|"uncertain"
    responsibility_evaluator_reasoning: str

    # ── Retry / HITL ──────────────────────────────────────────────
    perf_retry_count: int              # 0 or 1 (max 1)
    hitl_retry_count: int              # 0 or 1 (max 1)
    hitl_triggered: bool
    hitl_status: str                   # "pending"|"completed"|"skipped"
    human_feedback: Optional[str]      # correction text from human
    human_feedback_timestamp: Optional[str]

    # ── Final Decision ────────────────────────────────────────────
    final_decision: str                # "allow"|"block"|"edit"|"hitl"
    final_response: str                # what is actually sent to user
    block_reason: Optional[str]        # why blocked, with evidence
    block_evidence: Optional[List[str]]

    # ── Observability ─────────────────────────────────────────────
    langsmith_run_id: Optional[str]
    langsmith_feedback_logged: bool
    cost_history_saved: bool           # whether this run was saved to LangGraph Store
```

---

## 3. LANGGRAPH NODE & EDGE ARCHITECTURE (COMPLETE)

### 3A. Node Definitions

```
NODE NAME                       TYPE            DESCRIPTION
─────────────────────────────────────────────────────────────────────────
input_guardrail_node            Sequential      Regex+embedding+Groq LLM injection check
cache_check_node                Sequential      LiteLLM semantic cache lookup
rag_router_node                 Sequential      Load policy config; dispatch use-case
use_case_rag_agent              Sequential      Hybrid RAG (vector+BM25+KG) + generation
parallel_eval_fan_out           Fan-Out         LangGraph Send() → performance + responsibility
performance_ragas_node          Parallel        RAGAS library evaluation
performance_xgboost_node        Parallel        Feature extract + XGBoost predict
performance_entity_drift_node   Parallel        spaCy NER → entity drift score
performance_fan_in              Fan-In          Merge ragas+xgboost+entity drift scores
perf_llm_evaluator_node         Sequential      LLM interprets all perf scores → pass/retry/hitl
responsibility_rag_node         Parallel        Hybrid RAG over EU AI Act / NIST / ISO 42001
toxicity_fan_out                Fan-Out         Send() → 3 toxicity models simultaneously
toxicity_detoxify_node          Parallel        Detoxify('unbiased').predict(rag_answer)
toxicity_bert_node              Parallel        unitary/toxic-bert HuggingFace pipeline
toxicity_roberta_node           Parallel        s-nlp/roberta_toxicity_classifier pipeline
toxicity_fan_in                 Fan-In          Aggregate 3 toxicity scores
responsibility_fan_in           Fan-In          Merge resp_rag + toxicity_fan_in outputs
resp_llm_evaluator_node         Sequential      LLM: safe/unsafe/uncertain from all resp signals
eval_fan_in                     Fan-In          Merge performance + responsibility decisions
final_decision_node             Conditional     allow|block|hitl|perf_retry
hitl_interrupt_node             INTERRUPT       LangGraph interrupt(); await human input
hitl_resume_node                Sequential      Merge human feedback into updated_query
perf_retry_node                 Sequential      Generate correction; update updated_query; re-enter
langsmith_log_node              Sequential      Log full trace, costs, scores to LangSmith
cache_write_node                Sequential      Write successful response to Redis semantic cache
cost_history_node               Sequential      Write winning pattern to LangGraph Store
block_response_node             Terminal        Format block explanation + evidence
allow_response_node             Terminal        Format final answer; return to user
```

### 3B. Edges (Complete Graph)

```python
# Pseudocode — actual implementation uses StateGraph

graph = StateGraph(ControlPlaneState)

# ── Linear pre-RAG path ──────────────────────────────────────────────
graph.add_edge(START, "input_guardrail_node")
graph.add_conditional_edges(
    "input_guardrail_node",
    lambda s: "block" if not s["input_guardrail_passed"] else "cache",
    {"block": "block_response_node", "cache": "cache_check_node"}
)
graph.add_conditional_edges(
    "cache_check_node",
    lambda s: "allow" if s["cache_hit"] else "route",
    {"allow": "allow_response_node", "route": "rag_router_node"}
)
graph.add_edge("rag_router_node", "use_case_rag_agent")

# ── Fan-out to parallel eval ─────────────────────────────────────────
graph.add_conditional_edges(
    "use_case_rag_agent",
    lambda s: ["parallel_eval_fan_out"],   # always fan-out
    ["parallel_eval_fan_out"]
)
# parallel_eval_fan_out uses Send() to spawn two concurrent sub-paths:
# Send("performance_ragas_node", state)
# Send("performance_xgboost_node", state)
# Send("performance_entity_drift_node", state)
# Send("responsibility_rag_node", state)
# Send("toxicity_fan_out", state)

# ── Performance branch ───────────────────────────────────────────────
graph.add_edge("performance_ragas_node", "performance_fan_in")
graph.add_edge("performance_xgboost_node", "performance_fan_in")
graph.add_edge("performance_entity_drift_node", "performance_fan_in")
graph.add_edge("performance_fan_in", "perf_llm_evaluator_node")

# ── Toxicity sub-fan-out ─────────────────────────────────────────────
# toxicity_fan_out uses Send() to spawn 3 concurrent:
# Send("toxicity_detoxify_node", state)
# Send("toxicity_bert_node", state)
# Send("toxicity_roberta_node", state)
graph.add_edge("toxicity_detoxify_node", "toxicity_fan_in")
graph.add_edge("toxicity_bert_node", "toxicity_fan_in")
graph.add_edge("toxicity_roberta_node", "toxicity_fan_in")
graph.add_edge("toxicity_fan_in", "responsibility_fan_in")
graph.add_edge("responsibility_rag_node", "responsibility_fan_in")
graph.add_edge("responsibility_fan_in", "resp_llm_evaluator_node")

# ── Eval fan-in ──────────────────────────────────────────────────────
graph.add_edge("perf_llm_evaluator_node", "eval_fan_in")
graph.add_edge("resp_llm_evaluator_node", "eval_fan_in")

# ── Final decision (CONDITIONAL) ─────────────────────────────────────
graph.add_conditional_edges(
    "eval_fan_in",
    final_decision_router,   # function below
    {
        "allow": "allow_response_node",
        "block": "block_response_node",
        "hitl":  "hitl_interrupt_node",
        "perf_retry": "perf_retry_node",
    }
)

# ── Retry path ────────────────────────────────────────────────────────
graph.add_conditional_edges(
    "perf_retry_node",
    lambda s: "rag" if s["perf_retry_count"] <= 1 else "block",
    {"rag": "use_case_rag_agent", "block": "block_response_node"}
)

# ── HITL path ─────────────────────────────────────────────────────────
graph.add_edge("hitl_interrupt_node", "hitl_resume_node")
graph.add_conditional_edges(
    "hitl_resume_node",
    lambda s: "rag" if s["hitl_retry_count"] <= 1 else "block",
    {"rag": "use_case_rag_agent", "block": "block_response_node"}
)

# ── Terminal nodes ────────────────────────────────────────────────────
graph.add_edge("allow_response_node", "langsmith_log_node")
graph.add_edge("block_response_node", "langsmith_log_node")
graph.add_edge("langsmith_log_node", "cache_write_node")   # only write if allow
graph.add_edge("cache_write_node", "cost_history_node")
graph.add_edge("cost_history_node", END)
```

### 3C. `final_decision_router` Logic

```python
def final_decision_router(state: ControlPlaneState) -> str:
    resp_decision = state["responsibility_evaluator_decision"]
    perf_decision = state["performance_evaluator_decision"]

    # Responsibility UNSAFE → always block, never retry
    if resp_decision == "unsafe":
        return "block"

    # Responsibility UNCERTAIN → HITL (if not already retried)
    if resp_decision == "uncertain" and state["hitl_retry_count"] < 1:
        return "hitl"

    # Performance RETRY → retry RAG (if not already retried)
    if perf_decision == "retry" and state["perf_retry_count"] < 1:
        return "perf_retry"

    # Both safe and acceptable → allow
    if resp_decision == "safe" and perf_decision == "pass":
        return "allow"

    # All other ambiguous/exhausted cases → HITL, then block
    if state["hitl_retry_count"] < 1:
        return "hitl"
    return "block"
```

---

## 4. AGENT-BY-AGENT SPECIFICATION

### 4.1 Input Guardrail Node

| Field | Value |
|---|---|
| **Input** | `original_query`, `use_case` |
| **Processing** | ① Regex patterns for prompt injection (ignore previous, forget instructions, etc.) ② Sentence embedding of query → cosine similarity to known jailbreak embeddings (pre-built index) ③ Groq `llama-3.1-8b-instant` (free-tier) for LLM-based jailbreak classification ④ Microsoft Presidio for PII scan |
| **Model** | Groq `llama-3.1-8b-instant` (LiteLLM: `groq/llama-3.1-8b-instant`) |
| **Output** | `input_guardrail_passed: bool`, `guardrail_flags: List[str]` |
| **State Update** | `input_guardrail_passed`, `guardrail_flags` |
| **Next Node** | `block_response_node` if failed, else `cache_check_node` |
| **Latency target** | < 50ms (regex+presidio local; Groq call async) |

### 4.2 Semantic Cache Check Node

| Field | Value |
|---|---|
| **Input** | `updated_query` (or `original_query` on first pass) |
| **Processing** | LiteLLM semantic cache: embed query with `all-MiniLM-L6-v2` → query Redis+Qdrant for cosine similarity ≥ 0.95 |
| **Model** | `all-MiniLM-L6-v2` (local, sentence-transformers) |
| **Tool** | LiteLLM `cache_type="semantic"` + Redis + Qdrant |
| **Output** | `cache_hit: bool`, `cached_response: Optional[Dict]` |
| **State Update** | `cache_hit`, `cached_response` |
| **Next Node** | `allow_response_node` on hit, else `rag_router_node` |
| **Latency target** | < 20ms on hit |

### 4.3 RAG Router Node

| Field | Value |
|---|---|
| **Input** | `use_case`, `geography`, `updated_query` |
| **Processing** | Load YAML policy config for `use_case+geography`. Set `active_policy` in state. Determine which RAG sub-agent to activate. |
| **Model** | No LLM call — pure routing logic |
| **Output** | `active_policy: Dict` |
| **State Update** | `active_policy` |
| **Next Node** | `use_case_rag_agent` (same node, use_case determines knowledge source) |

### 4.4 Use-Case RAG Agent (Agentic RAG)

| Field | Value |
|---|---|
| **Input** | `updated_query`, `active_policy`, `conversation_history` |
| **Processing** | ① Hybrid retrieval: BM25 (`rank_bm25`) + Vector search (Chroma) → RRF merge ② KG traversal (NetworkX) for structured fact lookup ③ Cross-encoder context trimming: `cross-encoder/ms-marco-MiniLM-L-6-v2` → drop bottom 30% chunks ④ LLM generation with trimmed context |
| **Model** | `gemini/gemini-1.5-flash` via LiteLLM (medium model) |
| **Fallback** | `groq/llama-3.1-70b-versatile` → `groq/llama-3.1-8b-instant` |
| **Output** | `retrieved_context`, `retrieval_metadata`, `rag_answer`, `token_stats` |
| **State Update** | All RAG output fields |
| **Next Node** | `parallel_eval_fan_out` |

### 4.5 RAGAS Agent (Parallel — Performance Branch)

| Field | Value |
|---|---|
| **Input** | `updated_query`, `retrieved_context`, `rag_answer` |
| **Processing** | RAGAS library: `faithfulness`, `answer_relevancy`, `context_recall`, `context_precision`, `answer_correctness` |
| **Model** | RAGAS uses LLM as judge: route to `groq/llama-3.1-70b-versatile` (free-tier, sufficient for RAGAS judge) |
| **Fallback** | `gemini/gemini-1.5-flash` |
| **Output** | `ragas_scores: Dict[str, float]` |
| **State Update** | `ragas_scores` |
| **Next Node** | `performance_fan_in` |

### 4.6 XGBoost Agent (Parallel — Performance Branch)

| Field | Value |
|---|---|
| **Input** | `updated_query`, `retrieved_context`, `rag_answer` |
| **Processing** | ① `all-MiniLM-L6-v2` embed query+answer → cosine similarity (`overlap_score`) ② spaCy `en_core_web_sm` NER on both → `entity_consistency` ③ `cross-encoder/nli-deberta-v3-small` → `nli_contradiction_score` ④ `response_length` ⑤ Feed 4 features → pre-trained XGBoost (trained on RAGTruth) → `hallucination_prob` |
| **Model** | All local: sentence-transformers, spaCy, HuggingFace, XGBoost — **no LLM call** |
| **Output** | `xgboost_scores: {overlap_score, entity_consistency, nli_contradiction_score, response_length, hallucination_prob}` |
| **State Update** | `xgboost_scores` |
| **Next Node** | `performance_fan_in` |

### 4.7 Entity Drift Agent (Parallel — Performance Branch)

| Field | Value |
|---|---|
| **Input** | `retrieved_context`, `rag_answer`, `conversation_history` |
| **Processing** | ① spaCy NER on `retrieved_context` → source entities ② spaCy NER on `rag_answer` → response entities ③ Compute entity overlap and entities present in response but absent from context (hallucinated entities) ④ Compare entity set across conversation turns for drift |
| **Model** | spaCy `en_core_web_sm` (local, CPU) |
| **Output** | `entity_drift_results: {drifted_entities, drift_score, hallucinated_entities}` |
| **State Update** | `entity_drift_results` |
| **Next Node** | `performance_fan_in` |

### 4.8 Performance LLM Evaluator

| Field | Value |
|---|---|
| **Input** | `ragas_scores`, `xgboost_scores`, `entity_drift_results`, `token_stats`, `active_policy` |
| **Processing** | Structured prompt with all metrics → LLM reasons: is performance acceptable? If bad: generate correction instruction. Decision: `pass` / `retry` / `hitl`. |
| **Model** | `groq/llama-3.1-70b-versatile` (strong reasoning, free-tier) |
| **Fallback** | `gemini/gemini-1.5-flash` |
| **Output** | `performance_evaluator_decision`, `performance_evaluator_reasoning` |
| **State Update** | Both fields |
| **Next Node** | `eval_fan_in` |

### 4.9 Responsibility RAG Agent (Parallel — Responsibility Branch)

| Field | Value |
|---|---|
| **Input** | `updated_query`, `rag_answer`, `retrieved_context`, `geography` |
| **Processing** | ① Vector search on EU AI Act / NIST AI RMF / ISO 42001 / GDPR corpus → top-k clauses ② KG traversal: `Policy → RiskCategory → RequiredRemediation` ③ Threshold: only flag if retrieval similarity > 0.70 AND answer semantically matches the violating clause (NLI contradiction check against the clause) |
| **Threshold** | `retrieval_score ≥ 0.70` + `nli_entailment(rag_answer, clause) ≥ 0.60` |
| **Model** | `gemini/gemini-1.5-flash` for violation detection reasoning |
| **Fallback** | `groq/llama-3.1-70b-versatile` |
| **Output** | `resp_retrieved_laws`, `resp_retrieval_scores`, `resp_kg_path`, `responsibility_score`, `violated_law`, `evidence_chunks` |
| **State Update** | All responsibility RAG fields |
| **Next Node** | `responsibility_fan_in` |

### 4.10 Toxicity Agent — 3 Models in Parallel

#### 4.10a Detoxify Node
| Field | Value |
|---|---|
| **Input** | `rag_answer` |
| **Processing** | `Detoxify('unbiased').predict(rag_answer)` → 7 scores |
| **Model** | `detoxify` (local PyTorch, CPU/GPU) |
| **Output** | `detoxify_scores: Dict[str,float]`, `detoxify_label` |

#### 4.10b Toxic-BERT Node
| Field | Value |
|---|---|
| **Input** | `rag_answer` |
| **Processing** | HuggingFace pipeline `text-classification`, model `unitary/toxic-bert` |
| **Model** | `unitary/toxic-bert` (local HuggingFace, CPU/GPU) |
| **Output** | `toxic_bert_score: float`, `toxic_bert_label: str` |

#### 4.10c RoBERTa Toxicity Node
| Field | Value |
|---|---|
| **Input** | `rag_answer` |
| **Processing** | HuggingFace pipeline `text-classification`, model `s-nlp/roberta_toxicity_classifier` |
| **Model** | `s-nlp/roberta_toxicity_classifier` (local HuggingFace, CPU/GPU) |
| **Output** | `roberta_tox_score: float`, `roberta_tox_label: str` |

#### Toxicity Fan-In Aggregation
```python
def aggregate_toxicity(state):
    scores = [
        state["detoxify_scores"]["toxicity"],
        state["toxic_bert_score"],
        state["roberta_tox_score"],
    ]
    aggregate = max(scores)  # conservative: take worst-case
    flagged = aggregate >= 0.5  # threshold from active_policy
    return {
        "toxicity_aggregate": aggregate,
        "toxicity_flagged": flagged
    }
```

### 4.11 Responsibility LLM Evaluator

| Field | Value |
|---|---|
| **Input** | `resp_retrieved_laws`, `responsibility_score`, `violated_law`, `evidence_chunks`, `toxicity_aggregate`, `toxicity_flagged`, `detoxify_scores`, `toxic_bert_label`, `roberta_tox_label`, `active_policy` |
| **Processing** | Structured prompt: "Given the following toxicity scores and retrieved regulatory violations, determine: safe / unsafe / uncertain. If unsafe, cite the exact clause and evidence." |
| **Model** | `gemini/gemini-1.5-flash` (strong, for reliable legal reasoning) |
| **Fallback** | `groq/llama-3.1-70b-versatile` |
| **Output** | `responsibility_evaluator_decision`, `responsibility_evaluator_reasoning` |
| **State Update** | Both fields; if unsafe, populate `block_reason` and `block_evidence` |
| **Next Node** | `eval_fan_in` |

### 4.12 HITL Interrupt Node

| Field | Value |
|---|---|
| **Input** | Full state snapshot |
| **Processing** | `interrupt(value=hitl_payload)` — LangGraph native interrupt. Execution halts. Presents to human: `{rag_answer, retrieved_context, performance_evaluator_reasoning, responsibility_evaluator_reasoning, toxicity_aggregate, violated_law, evidence_chunks}`. Human provides: `human_feedback` (correction or approval text). |
| **LangGraph API** | `interrupt()` built-in function (LangGraph ≥ 0.2) + `Command(resume=human_input)` to resume |
| **State Update** | `hitl_status="pending"` on interrupt; `hitl_status="completed"`, `human_feedback`, `hitl_retry_count += 1` on resume |
| **Next Node** | `hitl_resume_node` after human submits |

### 4.13 Performance Retry Node

| Field | Value |
|---|---|
| **Input** | `updated_query`, `performance_evaluator_reasoning` |
| **Processing** | Append performance correction instruction to `updated_query`. Increment `perf_retry_count`. |
| **Model** | No LLM call — string operation |
| **Output** | `updated_query` (enhanced), `perf_retry_count` |
| **Next Node** | `use_case_rag_agent` (re-enters RAG with improved query) |

---

## 5. DATA SOURCES (Concrete, Verified, Specific)

### 5A. RAG Agent Knowledge Bases

#### 5A-1. HR Policy Agent
| Field | Detail |
|---|---|
| **Source** | SHRM HR Policy Templates (member-accessible, free tier available): https://www.shrm.org/topics-tools/tools/policies ✅ VERIFIED |
| **Also use** | U.S. Department of Labor — FMLA, ADA, FLSA policy documents: https://www.dol.gov/agencies/whd ✅ VERIFIED |
| **Also use** | UK ACAS Code of Practice on Disciplinary and Grievance Procedures (free PDF): https://www.acas.org.uk/acas-code-of-practice-on-disciplinary-and-grievance-procedures ✅ VERIFIED |
| **Format** | PDF → chunk at 512 tokens, overlap 64 |
| **Why appropriate** | Real HR policy text = realistic HR query/answer pairs. Covers leave, compensation, disciplinary, DEI. |
| **Questions enabled** | "What is our parental leave policy?", "Can an employee appeal a termination?", "What constitutes workplace harassment?" |
| **Ingestion** | `PyPDF2` + `langchain.text_splitter.RecursiveCharacterTextSplitter` → embed with `all-MiniLM-L6-v2` → Chroma |
| **Demo mix** | Ask HR questions; show policy clause retrieval; show entity drift when answer hallucinates a number |

#### 5A-2. Customer Support Tool
| Field | Detail |
|---|---|
| **Source** | Ubuntu Dialogue Corpus (customer support conversations): https://github.com/rkadlec/ubuntu-ranking-dataset-creator ✅ VERIFIED |
| **Also use** | Bitext Customer Support Intent Dataset (HuggingFace): https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset ✅ VERIFIED |
| **Also use** | Amazon Q&A Dataset (HuggingFace — sentence-transformers mirror, ~1.4M pairs, replaces dead SNAP link): https://huggingface.co/datasets/sentence-transformers/amazon-qa ✅ VERIFIED |
| **Format** | Parquet/JSONL → extract Q&A pairs as documents |
| **Why appropriate** | Real customer service conversations; spans billing, shipping, returns, account issues |
| **Questions enabled** | "How do I return a product?", "My account is locked", "Where is my order?" |
| **Ingestion** | Convert Q&A pairs to `Document(page_content=q+a, metadata={intent, category})` → Chroma |

#### 5A-3. Internal Knowledge Assistant
| Field | Detail |
|---|---|
| **Source** | Stack Exchange Data Dump — IT/SoftwareEngineering: https://archive.org/details/stackexchange |
| **Also use** | Microsoft Docs (open source on GitHub): https://github.com/MicrosoftDocs/azure-docs |
| **Also use** | Confluence-style mock internal wiki (generate 50 pages for demo using GPT-4o-mini) |
| **Why appropriate** | Simulates a real internal KB mixing technical docs, IT procedures, and process guides |
| **Questions enabled** | "How do I reset VPN access?", "What is the code review process?", "Which team owns the payment service?" |

#### 5A-4. Decision Support Tool
| Field | Detail |
|---|---|
| **Source** | RAGTruth dataset (primary): https://github.com/ParticleMedia/RAGTruth — use the source documents (news/Wikipedia) as decision context ✅ VERIFIED |
| **Also use** | SEC EDGAR Full-Text Search API (free, no key needed): https://efts.sec.gov/LATEST/search-index?q=%22annual+report%22&forms=10-K ✅ VERIFIED — use `https://data.sec.gov/submissions/CIK{number}.json` to pull filings programmatically |
| **Also use** | World Bank Open Data for macro-economic decision context: https://data.worldbank.org/ ✅ VERIFIED |
| **Why appropriate** | Structured factual context for data-backed decision questions; mixes loosely and well-governed sources |
| **Questions enabled** | "Should we expand into this market?", "What are the risk factors for this investment?", "Summarize the Q3 performance indicators." |

### 5B. Responsibility RAG Knowledge Base (EU AI Act + NIST + ISO)

| Source | URL | What it Covers | Why Needed |
|---|---|---|---|
| EU AI Act (full text) | https://artificialintelligenceact.eu/the-act/ ✅ VERIFIED | High-risk AI, transparency, data governance, prohibited practices | Primary European regulatory source |
| NIST AI RMF 1.0 (PDF, direct download) | https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf ✅ VERIFIED (old `airc.nist.gov/RMF` → 404) | Govern, Map, Measure, Manage functions | US framework for AI risk |
| NIST AI RMF Landing Page | https://www.nist.gov/itl/ai-risk-management-framework ✅ VERIFIED | All RMF resources, playbook, crosswalks | Supplementary resource |
| ISO/IEC 42001:2023 | ISO website / arXiv preprint: https://arxiv.org/abs/2402.01555 | AI management system standard | International standard |
| GDPR (full text) | https://gdpr-info.eu/ ✅ VERIFIED | PII, data processing, right to explanation | Data protection regulation |
| OECD AI Principles | https://oecd.ai/en/ai-principles ✅ VERIFIED | Accountability, transparency, safety | Supranational principles |

**Knowledge Graph construction:**
```
Node types: Policy, Article, RiskCategory, RequiredRemediation, UseCase
Edges: Policy --contains--> Article
       Article --addresses--> RiskCategory
       RiskCategory --requires--> RequiredRemediation
       UseCase --subject_to--> Article

Example:
EU_AI_Act --contains--> Article_10
Article_10 --addresses--> DataGovernance
DataGovernance --requires--> DataAudit+PIIRedaction
CustomerSupport --subject_to--> Article_10
```
**Build with:** NetworkX (demo) or Neo4j Community Edition (production)

### 5C. Toxicity/Harmful Datasets (For Judge Demonstrations)

| Dataset | Source | Contains | Integration | Expected Demo Behavior |
|---|---|---|---|---|
| **Jigsaw Unintended Bias** | https://kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification | 1.8M comments, toxicity + identity labels | Used to train XGBoost bias classifier; demo harmful comment injection → system blocks | All 3 toxicity models fire; resp evaluator blocks; shows evidence |
| **RAGTruth** | https://github.com/ParticleMedia/RAGTruth | 18K LLM responses with hallucination span labels | Pre-build XGBoost on it; demo a hallucinated response → system flags | XGBoost hallucination_prob high; entity drift fires; perf evaluator retries |
| **ToxiGen** | https://github.com/microsoft/ToxiGen | 274K toxic/benign statements targeting 13 groups | Use as adversarial test set for toxicity agent | Detoxify + BERT + RoBERTa all score high; blocked |
| **AdvGLUE** | https://adversarialglue.github.io/ | Adversarial NLP samples including jailbreaks | Use as adversarial input guardrail test set | Input guardrail catches before RAG |
| **HarmBench** | https://github.com/centerforaisafety/HarmBench | 400 harmful behaviors across 7 categories | Feed as user queries → show system blocks pre-RAG | Input guardrail + resp evaluator both fire |

---

## 6. LITELLM MODEL ROUTING TABLE

Every agent LLM call routes through LiteLLM. The routing logic classifies task complexity and selects the cheapest model that reliably handles it.

| # | Agent / Task | Complexity | Primary Model | Fallback 1 | Fallback 2 | Reason |
|---|---|---|---|---|---|---|
| 1 | Input Guardrail — jailbreak classify | Low | `groq/llama-3.1-8b-instant` (free) | `groq/gemma2-9b-it` (free) | Local `Qwen/Qwen3-0.6B` via Ollama | Binary classification; fast; no reasoning depth needed |
| 2 | RAG Answer Generation — HR Policy | Medium | `gemini/gemini-1.5-flash` | `groq/llama-3.1-70b-versatile` | `groq/llama-3.1-8b-instant` | Good comprehension; flash is cheap + fast |
| 3 | RAG Answer Generation — Customer Support | Medium | `groq/llama-3.1-70b-versatile` (free) | `gemini/gemini-1.5-flash` | `groq/llama-3.1-8b-instant` | Free-tier; good for conversational |
| 4 | RAG Answer Generation — Internal KB | Low-Medium | `groq/gemma2-9b-it` (free) | `groq/llama-3.1-8b-instant` | Local `Qwen/Qwen3-4B` via Ollama | Simple lookup queries |
| 5 | RAG Answer Generation — Decision Support | High | `gemini/gemini-1.5-pro` | `gemini/gemini-1.5-flash` | `groq/llama-3.1-70b-versatile` | Requires multi-step reasoning |
| 6 | RAGAS LLM-as-judge | Medium | `groq/llama-3.1-70b-versatile` (free) | `gemini/gemini-1.5-flash` | `groq/llama-3.1-8b-instant` | Evaluation task; quality matters but cost sensitive |
| 7 | Performance LLM Evaluator | Medium | `groq/llama-3.1-70b-versatile` (free) | `gemini/gemini-1.5-flash` | `groq/llama-3.3-70b-versatile` | Interpretation task; free-tier sufficient |
| 8 | Responsibility RAG violation detection | High | `gemini/gemini-1.5-flash` | `groq/llama-3.1-70b-versatile` | `gemini/gemini-1.5-pro` | Legal clause matching needs good semantic understanding |
| 9 | Responsibility LLM Evaluator | High | `gemini/gemini-1.5-flash` | `groq/llama-3.1-70b-versatile` | `gemini/gemini-1.5-pro` | Final safety gate; must be reliable |
| 10 | HITL — summary formatting for human | Low | `groq/llama-3.1-8b-instant` (free) | `groq/gemma2-9b-it` | Local Qwen3-4B | Formatting only |
| 11 | Perf Retry — correction instruction | Low | `groq/llama-3.1-8b-instant` (free) | `groq/gemma2-9b-it` | Local Qwen3-4B | String augmentation only |
| 12 | Cost anomaly explanation (optional) | Low | Local `Qwen/Qwen3-4B` via Ollama | `groq/llama-3.1-8b-instant` | — | Completely local; no API cost |

### Additional Free/Local Models to Add

| Model | Provider | Context | Best For | Access |
|---|---|---|---|---|
| `Qwen/Qwen3-0.6B` | Ollama (local) | 32K | Ultra-fast local classify | `ollama pull qwen3:0.6b` |
| `Qwen/Qwen3-4B` | Ollama (local) | 32K | Local medium tasks | `ollama pull qwen3:4b` |
| `google/gemma-3-1b-it` | Ollama / HuggingFace | 8K | Local lightweight generation | Free |
| `google/gemma-3-4b-it` | Ollama / HuggingFace | 128K | Local medium generation | Free |
| `groq/gemma2-9b-it` | Groq (free-tier) | 8K | Fast free-tier reasoning | Groq free API |
| `groq/llama-3.1-8b-instant` | Groq (free-tier) | 128K | Fast classification + simple gen | Groq free API |
| `groq/llama-3.1-70b-versatile` | Groq (free-tier) | 128K | Strong reasoning, free | Groq free API |
| `groq/llama-3.3-70b-versatile` | Groq (free-tier) | 128K | Latest Llama, strong | Groq free API |
| `groq/mistral-saba-24b` | Groq (free-tier) | 32K | Multilingual, safety | Groq free API |
| `gemini/gemini-1.5-flash` | Google AI Studio | 1M | Cheap, fast, large context | Free quota |
| `gemini/gemini-2.0-flash` | Google AI Studio | 1M | Newer, fast, large context | Free quota |
| `ollama/phi4-mini` | Ollama (local) | 128K | Microsoft small but smart | `ollama pull phi4-mini` |
| `ollama/deepseek-r1:1.5b` | Ollama (local) | 64K | Reasoning, tiny | `ollama pull deepseek-r1:1.5b` |
| `ollama/mistral:7b` | Ollama (local) | 8K | General local gen | `ollama pull mistral:7b` |

### LiteLLM Router Configuration

```python
from litellm import Router

router = Router(
    model_list=[
        {
            "model_name": "guardrail-classifier",
            "litellm_params": {"model": "groq/llama-3.1-8b-instant", "api_key": GROQ_KEY},
        },
        {
            "model_name": "guardrail-classifier",
            "litellm_params": {"model": "groq/gemma2-9b-it", "api_key": GROQ_KEY},
        },
        {
            "model_name": "guardrail-classifier",
            "litellm_params": {"model": "ollama/qwen3:0.6b", "api_base": "http://localhost:11434"},
        },
        # ... repeat for each task group
    ],
    fallbacks=[
        {"guardrail-classifier": ["groq/gemma2-9b-it", "ollama/qwen3:0.6b"]}
    ],
    routing_strategy="least-busy",   # minimizes latency
    num_retries=2,
    timeout=10,
)
```

---

## 7. LITELLM CACHING ARCHITECTURE

### Cache Levels (in order of application)

| Level | Type | Backend | What is Cached | Key Strategy | Similarity |
|---|---|---|---|---|---|
| **L1 — Semantic Query Cache** | Semantic | Redis + Qdrant | Full `{rag_answer, scores, evidence}` for sufficiently similar queries | Embedding of `updated_query` | Cosine similarity ≥ 0.95 |
| **L2 — Prompt/Provider Cache** | Exact / Prefix | LiteLLM native + provider KV cache | LLM generation calls with identical system prompt prefix | Hash of `system_prompt + trimmed_context` | Exact match (provider-level) |
| **L3 — Embedding Cache** | Exact | Redis | Embedding vectors for repeated chunks | Hash of chunk text | Exact |
| **L4 — BM25 Index Cache** | Disk | Local file | BM25 index per knowledge base | Per-KB identifier | N/A |

### Where Cache Node Belongs (relative to guardrails)

```
input_guardrail_node   ← BEFORE cache check
       │ (PASS)
cache_check_node       ← AFTER guardrail (critical: don't cache injected queries)
       │ (MISS)
rag_router_node
```

**Why guardrails run before cache:** A successful cache hit on a guardrail-failing query would serve a blocked query from cache. This is a security vulnerability. Guardrails must always run first.

**What is NOT cached:**
- HITL decisions (human-dependent)
- Responsibility evaluator decisions (regulation may change)
- Toxicity scores (must run live; scores depend on model state)

### LiteLLM Semantic Cache Setup

```python
import litellm
from litellm.caching import Cache

litellm.cache = Cache(
    type="redis-semantic",
    host="localhost",
    port=6379,
    password=REDIS_PASSWORD,
    similarity_threshold=0.95,
    embedding_model="text-embedding-3-small",  # or local all-MiniLM-L6-v2
)

# Enable caching for specific calls:
response = litellm.completion(
    model="gemini/gemini-1.5-flash",
    messages=[...],
    cache={"no-cache": False, "no-store": False},  # allow cache
)
```

**Latency impact:** L1 cache hit saves ~300-800ms (full RAG+eval pipeline). L2 provider prefix cache saves ~50-200ms on repeated system prompts.

---

## 8. GUARDRAIL MAP (Complete)

| # | Guardrail | Position in Graph | What it Checks | Failure Action | LangGraph Node/Edge |
|---|---|---|---|---|---|
| G1 | Regex Prompt Injection | `input_guardrail_node` | Known injection patterns: "ignore previous instructions", "forget your prompt", "DAN mode", "jailbreak" | Block immediately; set `guardrail_flags=["prompt_injection"]` | Conditional edge → `block_response_node` |
| G2 | Embedding Anomaly (Jailbreak) | `input_guardrail_node` | Cosine similarity to pre-built jailbreak embedding index (≥ 0.80 = flag) | Block; flag as `jailbreak_detected` | Same conditional edge |
| G3 | LLM Jailbreak Classifier | `input_guardrail_node` | `groq/llama-3.1-8b-instant` classifies query: "Is this a prompt injection or jailbreak attempt?" Binary label + confidence | Block if confidence ≥ 0.85 | Same conditional edge |
| G4 | PII Detection (Input) | `input_guardrail_node` | Microsoft Presidio scan for SSN, credit card, email, phone, address in `original_query` | Flag detected PII entities; sanitize query before proceeding | `guardrail_flags += ["pii_input"]`; sanitize and continue |
| G5 | Retrieval Safety | `use_case_rag_agent` (post-retrieval) | Check retrieved chunks for confidential markers; filter chunks flagged as out-of-scope for use_case | Drop flagged chunks from context | Internal to RAG node; log warning |
| G6 | Context Length / Cost Guard | `use_case_rag_agent` (pre-generation) | Total token count of context + query exceeds policy threshold | Context trimming agent fires; drop bottom 30% chunks | Internal trim logic |
| G7 | Output PII Detection | `allow_response_node` (pre-output) | Presidio scan on `rag_answer` before sending to user | Redact PII; log `guardrail_flags += ["pii_output"]` | Post-process in allow_response_node |
| G8 | Toxicity Gate | `resp_llm_evaluator_node` | `toxicity_aggregate ≥ active_policy["toxicity_threshold"]` | `responsibility_evaluator_decision = "unsafe"` → block | Conditional edge in `final_decision_node` |
| G9 | Regulatory Violation Gate | `resp_llm_evaluator_node` | `responsibility_score ≥ active_policy["responsibility_threshold"]` with matched clause | Block + return cited evidence | Same |
| G10 | Retry Limit Guard | `perf_retry_node` + `hitl_resume_node` | `perf_retry_count ≥ 1` or `hitl_retry_count ≥ 1` | Route to block instead of retry | Conditional edge routing |
| G11 | Tool-Call Safety (future) | Any tool-using agent node | Prevent agents from calling unauthorized external APIs | Block tool call; log | Not yet in scope for prototype, mark as future |

---

## 9. PARALLELISM MAP

### What Runs in Parallel

```
Level 1: Performance Branch ∥ Responsibility Branch
         (both receive {user_query, retrieved_context, rag_answer} simultaneously via Send())

Level 2 (within Performance Branch):
         RAGAS Agent ∥ XGBoost Agent ∥ Entity Drift Agent
         (all three receive the same state simultaneously via Send())

Level 3 (within Responsibility Branch):
         Responsibility RAG Agent ∥ Toxicity Fan-Out
         (RAG retrieval and toxicity scoring run simultaneously)

Level 4 (within Toxicity Fan-Out):
         Detoxify Node ∥ Toxic-BERT Node ∥ RoBERTa Toxicity Node
         (all three models score simultaneously — run async with asyncio.gather())

Other async operations:
         LangSmith logging runs as a non-blocking background write after response
         Cache write runs after allow decision, non-blocking
         Cost history write runs non-blocking
```

### What CANNOT Run in Parallel (and Why)

| Sequential Constraint | Reason |
|---|---|
| Guardrail → Cache | Guardrail must complete before cache check (security) |
| Cache → RAG | Cache hit short-circuits RAG; they are mutually exclusive |
| RAG → Eval Fan-Out | Eval requires RAG answer; cannot start before generation |
| Toxicity Fan-In → Responsibility Fan-In | Toxicity scores must be collected before responsibility evaluator can aggregate |
| Both evaluators → Eval Fan-In | Both must finish before final decision |
| Eval Fan-In → Final Decision | Final decision requires both evaluator results |
| Perf Retry → RAG (re-enter) | Must be sequential: generate correction → re-enter RAG |
| HITL Interrupt → Resume | Human response must be received before resuming |

---

## 10. RETRY ARCHITECTURE

### Performance Retry (Automatic, Max 1)

```
perf_llm_evaluator_node
        │ decision = "retry"
        ▼
perf_retry_node
  • perf_retry_count += 1
  • updated_query = f"{original_query}\n[SYSTEM CORRECTION: {perf_evaluator_reasoning}]"
        │
        ▼ (conditional: if perf_retry_count == 1)
use_case_rag_agent   ← re-enters full pipeline with improved query
        │
        ▼
parallel_eval_fan_out   ← second evaluation round
        │
eval_fan_in
        │ (conditional: if perf_retry_count >= 1, no more perf_retry allowed)
final_decision_node  ← can only go to allow, block, or hitl — not perf_retry again
```

### HITL Retry (Human-in-Loop, Max 1)

```
final_decision_node
        │ decision = "hitl"
        ▼
hitl_interrupt_node
  • interrupt(value={
        "rag_answer": state["rag_answer"],
        "retrieved_context": state["retrieved_context"][:3],
        "performance_reasoning": state["performance_evaluator_reasoning"],
        "responsibility_reasoning": state["responsibility_evaluator_reasoning"],
        "toxicity_aggregate": state["toxicity_aggregate"],
        "violated_law": state["violated_law"],
        "evidence": state["evidence_chunks"],
        "instruction": "Please review this response and provide a correction or approval."
    })
  • hitl_status = "pending"
  ← Execution halts here; LangGraph persists state to checkpointer (Redis/SQLite)

[Human reviews payload in UI]
[Human submits: human_feedback = "The answer should clarify X and not mention Y."]

  • Command(resume={"human_feedback": human_feedback}) sent to graph
        │
        ▼
hitl_resume_node
  • hitl_retry_count += 1
  • updated_query = f"{original_query}\n[HUMAN CORRECTION: {human_feedback}]"
  • hitl_status = "completed"
        │
        ▼ (conditional: if hitl_retry_count == 1)
use_case_rag_agent   ← re-enters pipeline with human-corrected query
        │
        ▼
parallel_eval_fan_out
        │
eval_fan_in
        │ (conditional: if hitl_retry_count >= 1, no more hitl allowed)
final_decision_node  ← can only go to allow or block
```

### Key Differences: Performance Retry vs HITL

| Aspect | Performance Retry | HITL |
|---|---|---|
| Trigger | `perf_evaluator_decision == "retry"` | `final_decision_node == "hitl"` |
| Who provides correction | System (LLM-generated) | Human reviewer |
| LangGraph mechanism | Normal edge re-entry | `interrupt()` + `Command(resume=...)` |
| State preserved by | LangGraph state carries forward | LangGraph checkpointer (persistent) |
| Max attempts | 1 | 1 |
| Happens when | Performance metrics below threshold | Ambiguous/uncertain/missing-info cases |
| Can trigger each other | Yes (perf retry first; HITL if still uncertain) | No (HITL is the final human gate) |

---

## 11. RESPONSIBILITY DECISION ARCHITECTURE

```
Responsibility RAG Agent outputs:
  • responsibility_score (float, computed as weighted combo of retrieval score + NLI match score)
  • violated_law (string or None)
  • evidence_chunks (list)

Toxicity Fan-In outputs:
  • toxicity_aggregate (float: max of 3 model scores)
  • toxicity_flagged (bool: aggregate ≥ policy threshold)

Both feed into Responsibility LLM Evaluator:

DECISION LOGIC:

if toxicity_flagged == True AND toxicity_aggregate >= 0.7:
    → UNSAFE (hard block, do not send to final decision)
    → populate block_reason with toxicity details

elif responsibility_score >= active_policy["resp_threshold"] AND violated_law is not None:
    → UNSAFE (regulatory violation)
    → populate block_reason with violated_law + evidence_chunks

elif 0.4 <= responsibility_score < active_policy["resp_threshold"]:
    → UNCERTAIN (possible issue, insufficient evidence)
    → trigger HITL

elif toxicity_aggregate >= 0.3 AND toxicity_aggregate < 0.7:
    → UNCERTAIN (low-confidence toxicity signal)
    → trigger HITL

else:
    → SAFE

UNSAFE path:
  block_response_node formats:
  {
    "decision": "block",
    "reason": block_reason,
    "violated_law": violated_law,
    "evidence": evidence_chunks,
    "toxicity_scores": {detoxify, toxic_bert, roberta},
    "recommendation": resp_kg_path["RequiredRemediation"]
  }

SAFE path → Continue to final_decision_node
UNCERTAIN path → HITL (human sees full evidence)
```

---

## 12. HUMAN-IN-THE-LOOP: LANGGRAPH IMPLEMENTATION

### LangGraph Interrupt Architecture (Concrete)

```python
# 1. Enable checkpointer (state persistence across interrupt)
from langgraph.checkpoint.redis import AsyncRedisSaver  # or SqliteSaver for dev
checkpointer = AsyncRedisSaver.from_conn_string("redis://localhost:6379")
graph = graph.compile(checkpointer=checkpointer, interrupt_before=["hitl_interrupt_node"])

# 2. HITL Interrupt Node
from langgraph.types import interrupt, Command

def hitl_interrupt_node(state: ControlPlaneState):
    hitl_payload = {
        "rag_answer": state["rag_answer"],
        "retrieved_context": state["retrieved_context"][:3],
        "performance_reasoning": state["performance_evaluator_reasoning"],
        "responsibility_reasoning": state["responsibility_evaluator_reasoning"],
        "toxicity_scores": {
            "aggregate": state["toxicity_aggregate"],
            "detoxify": state["detoxify_scores"],
            "toxic_bert": state["toxic_bert_score"],
            "roberta": state["roberta_tox_score"],
        },
        "violated_law": state["violated_law"],
        "evidence": state["evidence_chunks"],
        "instruction": "Review the AI response and provide a correction or type 'approve' to allow it.",
    }
    human_input = interrupt(value=hitl_payload)   # ← execution halts here
    return {
        "human_feedback": human_input,
        "hitl_status": "completed",
        "hitl_retry_count": state["hitl_retry_count"] + 1,
    }

# 3. Resume execution from UI/API
thread_id = "user-session-abc123"
config = {"configurable": {"thread_id": thread_id}}

# Run until interrupt
result = await graph.ainvoke({"original_query": query, ...}, config=config)

# Human reviews; submits feedback via UI
human_correction = "The answer should not mention employee names. Revise to be anonymous."

# Resume graph with human input
result = await graph.ainvoke(
    Command(resume=human_correction),
    config=config
)
# Graph resumes from hitl_interrupt_node with human_input = human_correction
```

### What the Human Sees (UI Payload)

```json
{
  "rag_answer": "John Smith's performance review shows...",
  "retrieved_context": ["HR Policy Section 3.2...", "..."],
  "performance_reasoning": "Entity consistency score 0.45 — some entities not found in source",
  "responsibility_reasoning": "Possible PII exposure: employee name mentioned without authorization",
  "toxicity_scores": {"aggregate": 0.12, "detoxify": {"toxicity": 0.01}, "toxic_bert": 0.08, "roberta": 0.12},
  "violated_law": "GDPR Article 5(1)(c) — data minimisation",
  "evidence": ["Personal data shall be adequate, relevant and limited to what is necessary..."],
  "instruction": "Review the AI response and provide a correction or type 'approve' to allow it."
}
```

---

## 13. COST ARCHITECTURE

### Total Cost Formula (Real-Time)

```
TC = [(N_in × P_in) + (N_out × P_out) + Cost_RAG] × (1 + R)

Where:
  N_in    = input tokens
  P_in    = price per input token (from LiteLLM token_cost_calculator)
  N_out   = output tokens
  P_out   = price per output token
  Cost_RAG = embedding API cost (if not local)
  R       = retry count (0 or 1)
```

### Cost Agent Node

The Cost Agent runs **within the performance branch** alongside RAGAS and XGBoost:

```python
def cost_agent_node(state: ControlPlaneState):
    token_stats = state["token_stats"]
    n_in = token_stats["input_tokens"]
    n_out = token_stats["output_tokens"]
    r = state["perf_retry_count"]
    
    # LiteLLM cost calculator
    from litellm import completion_cost
    tc = completion_cost(
        completion_response=state["litellm_response_object"],
    ) * (1 + r)
    
    # XGBoost Isolation Forest anomaly detection
    cost_features = np.array([[n_in, n_out, token_stats["latency_ms"], r]])
    anomaly_score = cost_isolation_forest.decision_function(cost_features)[0]
    cost_flag = "anomalous" if anomaly_score < -0.1 else "normal"
    
    return {
        "tc_score": tc,
        "cost_flag": cost_flag,
        "token_stats": token_stats,
    }
```

### LangSmith Dashboard Metrics

```
Per-query logged:
  • tc_score (total cost)
  • cost_flag (normal/anomalous)
  • latency_ms
  • ragas_scores
  • toxicity_aggregate
  • responsibility_score
  • final_decision
  • model_used
  • cache_hit

Aggregate dashboard (via LangSmith):
  • Average cost per successful query
  • Wasted spend % (retries + blocked responses)
  • Savings via cache hits
  • False positive rate (HITL overrides where human approved)
  • False negative rate (manual audit of allowed responses later flagged)
  • System trustworthiness score = 1 - FPR
```

---

## 14. LATENCY OPTIMIZATION STRATEGY

### Component-Level Latency Budget (~300ms total target)

| Component | Estimated Latency | Optimization |
|---|---|---|
| Input Guardrail (regex + presidio) | 5-15ms | Fully local, CPU |
| Input Guardrail (LLM jailbreak) | 30-80ms | Groq ultra-fast inference (~50ms) |
| Semantic Cache Check | 10-20ms | Redis in-memory lookup |
| RAG Routing | 1ms | Pure Python |
| RAG Retrieval (vector + BM25) | 20-50ms | Local Chroma + rank_bm25 |
| RAG KG Traversal | 5-10ms | In-memory NetworkX |
| Context Trimming (cross-encoder) | 10-30ms | Local MiniLM, CPU |
| RAG Generation (flash model) | 100-300ms | Gemini Flash or Groq (~100ms) |
| **RAGAS ∥ XGBoost ∥ Entity Drift** | **50-100ms** | **All parallel; XGBoost+entity drift fully local** |
| **Resp RAG ∥ Toxicity (3 models ∥)** | **80-200ms** | **Parallel; toxicity models local on GPU** |
| Perf LLM Evaluator | 80-150ms | Groq fast inference |
| Resp LLM Evaluator | 80-150ms | Gemini Flash |
| Final Decision Node | 1ms | Pure Python logic |
| Total (critical path) | ~200-400ms | With parallelism |

### vLLM Recommendation

Use **vLLM** for:
- Running toxicity models (detoxify + toxic-bert + roberta) with GPU batching if co-located
- Running Qwen3-4B or Gemma-3-4B locally as a fast generation fallback

```bash
# Start vLLM server for local model
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-4B \
  --port 8000 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.5

# Register in LiteLLM router
{"model_name": "local-medium", "litellm_params": {"model": "openai/Qwen3-4B", "api_base": "http://localhost:8000/v1"}}
```

**Trade-off table:**

| Option | Latency | Cost | Quality | GPU Needed | Best For |
|---|---|---|---|---|---|
| Local (Ollama CPU) | 500-2000ms | $0 | Low-Medium | No | Development |
| Local (vLLM GPU) | 50-200ms | $0 | Medium-High | Yes | Production local |
| Groq API (free) | 50-150ms | $0 (rate limited) | High | No | Low-volume prod |
| Gemini Flash | 100-300ms | ~$0.075/1M tokens | High | No | Default medium |
| Gemini Pro | 200-600ms | ~$1.25/1M tokens | Very High | No | Complex tasks only |

### Async Execution

All LangGraph nodes should be implemented as `async def` nodes using `asyncio`:

```python
async def use_case_rag_agent(state: ControlPlaneState):
    retrieval_task = asyncio.create_task(hybrid_retrieve(state["updated_query"]))
    # ... concurrent operations within node
    context = await retrieval_task
    answer = await litellm_call(context)
    return {...}
```

---

## 15. IMPLEMENTATION FEASIBILITY VERIFICATION

| Component | Library/Tool | Feasibility Status | Notes |
|---|---|---|---|
| LangGraph orchestration | `langgraph>=0.2` | ✅ Fully supported | `Send()`, `interrupt()`, `Command(resume=)` all in 0.2+ |
| LangChain RAG | `langchain>=0.3` | ✅ Fully supported | Text splitters, retrievers, chains |
| LiteLLM routing | `litellm>=1.40` | ✅ Fully supported | Router, fallbacks, semantic cache, cost calc |
| Chroma vector DB | `chromadb>=0.5` | ✅ Fully supported | Local or server mode |
| BM25 retrieval | `rank_bm25` | ✅ Fully supported | Pure Python |
| RAGAS evaluation | `ragas>=0.2` | ✅ Fully supported | LLM-as-judge; `faithfulness`, `answer_relevancy`, etc. |
| XGBoost classifier | `xgboost>=2.0` | ✅ Fully supported | Standard tabular ML |
| spaCy NER | `spacy>=3.7` + `en_core_web_sm` | ✅ Fully supported | `python -m spacy download en_core_web_sm` |
| Entity drift | Custom Python using spaCy | ✅ Implementable | ~20 lines of Python |
| Detoxify | `detoxify` | ✅ Fully supported | PyPI; PyTorch models |
| unitary/toxic-bert | HuggingFace `transformers` | ✅ Fully supported | `pipeline("text-classification", model="unitary/toxic-bert")` |
| s-nlp/roberta toxicity | HuggingFace `transformers` | ✅ Fully supported | `pipeline("text-classification", model="s-nlp/roberta_toxicity_classifier")` |
| Knowledge Graph | `networkx>=3.0` | ✅ Feasible for prototype | NetworkX in-memory; Neo4j for scale |
| Prompt caching | LiteLLM `cache_type="redis-semantic"` | ✅ Fully supported | Requires Redis + Qdrant |
| vLLM local inference | `vllm>=0.5` | ✅ Supported (GPU only) | CPU fallback via Ollama |
| Microsoft Presidio | `presidio-analyzer` | ✅ Fully supported | PII detection |
| LangSmith tracing | `langsmith` SDK | ✅ Fully supported | `LANGCHAIN_TRACING_V2=true` |
| HITL interrupt | LangGraph `interrupt()` | ✅ Supported since 0.2 | Requires checkpointer |
| Redis checkpointer | `langgraph-checkpoint-redis` | ✅ Available | For HITL state persistence |
| Groq free-tier | `groq` via LiteLLM | ✅ Available | Rate limits: 30 req/min free |
| Gemini Flash | `google-generativeai` via LiteLLM | ✅ Available | Free quota: 1M tokens/day |

---

## 16. STEP-BY-STEP IMPLEMENTATION GUIDE

### Phase 1 — Environment Setup (Day 1)
```bash
pip install langgraph langchain langchain-community litellm chromadb \
    rank-bm25 ragas xgboost spacy networkx detoxify transformers \
    presidio-analyzer presidio-anonymizer langsmith sentence-transformers \
    pdfplumber requests datasets

python -m spacy download en_core_web_sm
ollama pull qwen3:4b
ollama pull phi4-mini
```

### Phase 2 — Data Ingestion (Day 1-2)
1. Download RAGTruth from GitHub → extract features → train XGBoost classifier
2. Download HR policy PDFs → chunk → embed → ingest into Chroma collection `hr_policy`
3. Download Bitext customer support dataset → ingest into Chroma collection `customer_support`
4. Download EU AI Act PDF → chunk → embed → ingest into Chroma collection `eu_regulations`
5. Build NetworkX KG: `Policy → Article → RiskCategory → Remediation` nodes

### Phase 3 — Build LangGraph Nodes (Day 2-4)
- Implement each node as an `async def` function
- Test each node in isolation with mock state
- Wire conditional edges
- Test full graph with `graph.ainvoke(test_state)`

### Phase 4 — LiteLLM Router (Day 3)
- Configure router with all models and fallbacks
- Implement semantic cache with Redis
- Test fallback behavior by temporarily breaking primary model

### Phase 5 — Toxicity Models (Day 3)
- Load all 3 models; test `asyncio.gather()` concurrent scoring
- Verify aggregation logic with ToxiGen test samples

### Phase 6 — Evaluation Pipeline (Day 4)
- Integrate RAGAS with LiteLLM judge model
- Train XGBoost on RAGTruth (target: AUC > 0.75)
- Test entity drift detection

### Phase 7 — HITL + Dashboard (Day 5)
- Set up Redis checkpointer
- Build minimal Flask/FastAPI UI to receive interrupt payloads and send `Command(resume=...)`
- Connect LangSmith tracing; verify all metrics logged

### Phase 8 — Demo Preparation (Day 6)
- Load ToxiGen + HarmBench adversarial samples → demonstrate blocking
- Load hallucinated RAGTruth samples → demonstrate performance retry
- Load GDPR-violating responses → demonstrate responsibility block
- Record full end-to-end flow for competition video

---

## 17. ARCHITECTURE DIAGRAM COMPONENTS (For Competition Diagram)

Your competition diagram should show these **labeled boxes** and **arrows**:

```
[USER / API INPUT]
        ↓
[INPUT GUARDRAIL] — (Regex + Embedding + Groq LLM + Presidio)
        ↓ PASS / ← BLOCK
[SEMANTIC CACHE] — (LiteLLM Redis+Qdrant, ≥0.95 similarity)
        ↓ MISS / ← HIT → [CACHED RESPONSE]
[RAG ROUTER] — (Policy Config Loader, Use-Case Dispatcher)
        ↓
[AGENTIC RAG AGENT] — (Vector Search + BM25 + KG + Cross-Encoder Trim + LiteLLM Gen)
        ↓
    ╔═══════════════ PARALLEL FAN-OUT ════════════════╗
    ║                                                  ║
[PERFORMANCE BRANCH]                    [RESPONSIBILITY BRANCH]
    ║                                                  ║
 ┌──┴──────────────────┐          ┌────────────────────┴────────┐
 │  RAGAS ∥ XGBoost ∥  │          │ Resp. RAG Agent (EU/NIST/ISO)│
 │  Entity Drift       │          │          ∥                   │
 └──────────┬──────────┘          │ Toxicity: Detoxify ∥         │
            │                     │ Toxic-BERT ∥ RoBERTa         │
  [Perf LLM Evaluator]            └──────────────────┬──────────┘
            │                               [Resp LLM Evaluator]
            ╚═══════════════ FAN-IN ═════════════════╝
                                    ↓
                          [FINAL DECISION NODE]
                      ┌───────┬──────────┬──────┐
                    ALLOW   BLOCK      HITL  PERF-RETRY
                      ↓       ↓         ↓        ↓
                 [Answer] [Block+  [Human   [Retry RAG
                  to User  Evidence] Review]  max 1x]
                      ↓       ↓         ↓
                 [LangSmith Log + Cache Write + Cost History]
```

**Annotate with:**
- All parallel branches with `∥` symbols
- LiteLLM model for each LLM call
- Dataset names (RAGTruth, Jigsaw, EU AI Act, NIST)
- Max retry counts (Perf: ≤1, HITL: ≤1)
- "~300ms target latency"
- "LangSmith observability" annotation on all nodes

---

## OPEN QUESTIONS / DECISIONS FOR YOU

> [!IMPORTANT]
> **Decision 1 — GPU availability:** Do you have a GPU for the competition prototype? This determines whether vLLM and local model inference is practical, or whether Groq+Gemini free-tier is the primary path.

> [!IMPORTANT]
> **Decision 2 — HITL UI:** Do you want a minimal web UI (FastAPI + plain HTML) for the human review panel, or is a CLI mock sufficient for the demo?

> [!IMPORTANT]
> **Decision 3 — Redis availability:** Do you have Redis running locally, or should we use `SqliteSaver` as the checkpointer and `DiskCache` instead of Redis for the prototype?

> [!NOTE]
> **Implementation note:** The Cost Agent (Isolation Forest anomaly model) requires historical query logs to train. For the prototype demo, train it on synthetic `{n_in, n_out, latency}` samples or the RAGTruth response lengths as a proxy.

> [!NOTE]
> **Implementation note:** The Knowledge Graph for responsibility (EU AI Act) can be built manually in ~2 hours as a NetworkX graph with ~30 nodes. A pre-built JSON representation is sufficient for the prototype; full Neo4j is only needed for scale.
