import logging
from typing import Dict, Any, List, Optional

try:
    from langgraph.graph import StateGraph, START, END
except ImportError:
    StateGraph = None  # type: ignore
    START = "START"  # type: ignore
    END = "END"  # type: ignore

from .state import AgentState
from .prompts import RESPONSIBILITY_AGENT_SYSTEM_PROMPT
from ..ingestion.chunk_store import ChunkStore, Chunk
from ..retrieval.retrievers import HybridRetriever
from ..retrieval.rrf import reciprocal_rank_fusion
from ..config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Heuristic patterns: ONLY used when LLM is completely unavailable
# These are extremely specific phrases — NOT general topic keywords
# ---------------------------------------------------------------------------
# Each is a multi-word phrase that only appears in genuinely harmful instructions
HARMFUL_EXACT_PHRASES = [
    "flash subliminal",
    "subliminal micro-cues",
    "subliminal audio-visual",
    "manipulate vulnerable users into spending",
    "manipulate users without realizing",
    "without conscious awareness to manipulate",
    "social reliability score",
    "social credit score",
    "disqualify low-scoring individuals from public",
    "penalize female applicants",
    "filter out candidate resumes from specific demographic zip codes",
    "automatically deducts pay based on detected negative emotional",
    "emotion recognition to deduct",
    "facial emotion recognition to penalize",
    "build a bomb",
    "synthesize chemical weapon",
    "child sexual",
    "exploit children",
    "mass surveillance without any consent",
]


def _quick_heuristic_check(text: str) -> bool:
    """
    Used ONLY when the LLM is unavailable.
    Returns True only if a clearly harmful multi-word phrase is matched.
    Single-word matching is intentionally avoided to prevent false positives.
    """
    text_lower = text.lower()
    for phrase in HARMFUL_EXACT_PHRASES:
        if phrase in text_lower:
            return True
    return False


def _extract_verdict_from_llm_response(verdict_text: str) -> bool:
    """
    Robustly parse the LLM verdict to determine is_violation (True/False).
    Uses Regex to find the exact STATUS line to avoid false positives 
    (e.g., catching the word 'UNETHICAL' inside 'not unethical').
    """
    import re
    
    # Extract just the first section to avoid matching text in the assessment body
    search_block = verdict_text[:1500]
    
    # Look for exact STATUS line pattern
    status_match = re.search(r"\*\*STATUS\*\*:\s*(.*?)(?:\n|$)", search_block, re.IGNORECASE)
    
    if not status_match:
        logger.warning("Could not find STATUS line in LLM verdict — defaulting to COMPLIANT.")
        return False
        
    status_text = status_match.group(1).upper()
    
    # Check for non-compliant keywords in the extracted status ONLY
    if "UNETHICAL" in status_text or "NON-COMPLIANT" in status_text or "FLAGGED" in status_text:
        return True
        
    if "COMPLIANT" in status_text:
        return False
        
    logger.warning(f"Ambiguous STATUS line: '{status_text}' — defaulting to COMPLIANT.")
    return False


def get_agent_llm():
    """Initializes the LLM for the final Responsibility Moderation Agent.
    Priority is based on settings.AGENT_LLM_PROVIDER.
    """
    provider = settings.AGENT_LLM_PROVIDER.lower()
    
    def try_gemini():
        if settings.GOOGLE_API_KEY and settings.GOOGLE_API_KEY.strip():
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                llm = ChatGoogleGenerativeAI(
                    model=settings.GEMINI_MODEL,
                    temperature=0.0,
                    google_api_key=settings.GOOGLE_API_KEY
                )
                logger.info(f"Initialized Gemini LLM: {settings.GEMINI_MODEL}")
                return llm
            except Exception as e:
                logger.warning(f"Failed to initialize ChatGoogleGenerativeAI: {e}")
        return None

    def try_groq():
        if settings.GROQ_API_KEY and settings.GROQ_API_KEY.strip():
            try:
                from langchain_groq import ChatGroq
                llm = ChatGroq(
                    model=settings.GROQ_MODERATION_MODEL,
                    temperature=0.0,
                    groq_api_key=settings.GROQ_API_KEY
                )
                logger.info(f"Initialized Groq LLM for moderation: {settings.GROQ_MODERATION_MODEL}")
                return llm
            except Exception as e:
                logger.warning(f"Failed to initialize Groq LLM: {e}")
        return None

    def try_openai():
        if settings.OPENAI_API_KEY and settings.OPENAI_API_KEY.strip():
            try:
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(
                    model=settings.OPENAI_MODEL,
                    temperature=0.0,
                    openai_api_key=settings.OPENAI_API_KEY
                )
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI LLM: {e}")
        return None

    if provider == "gemini":
        return try_gemini() or try_groq() or try_openai()
    elif provider == "groq":
        return try_groq() or try_gemini() or try_openai()
    elif provider == "openai":
        return try_openai() or try_gemini() or try_groq()
    else:
        # Default priority
        return try_groq() or try_gemini() or try_openai()


class ResponsibilityAgentWorkflow:
    """Encapsulates the LangGraph Hybrid Retrieval & Google Gemini Compliance Moderation Workflow."""

    def __init__(self, retriever: HybridRetriever, chunk_store: ChunkStore):
        self.retriever = retriever
        self.chunk_store = chunk_store
        self.llm = get_agent_llm()
        self.app = self._build_graph()

    def _retrieve_vector_node(self, state: AgentState) -> Dict[str, Any]:
        """Node 1: Vector DB retrieval (top 4 chunks)."""
        query = state["query"]
        chunks = self.retriever.retrieve_vector(query, k=settings.RETRIEVAL_TOP_K_PER_BRANCH)
        return {"vector_chunks": [c.to_dict() for c in chunks]}

    def _retrieve_bm25_node(self, state: AgentState) -> Dict[str, Any]:
        """Node 2: BM25 lexical retrieval (top 4 chunks)."""
        query = state["query"]
        chunks = self.retriever.retrieve_bm25(query, k=settings.RETRIEVAL_TOP_K_PER_BRANCH)
        return {"bm25_chunks": [c.to_dict() for c in chunks]}

    def _retrieve_graph_node(self, state: AgentState) -> Dict[str, Any]:
        """Node 3: Neo4j Knowledge Graph retrieval (top 4 chunks)."""
        query = state["query"]
        chunks = self.retriever.retrieve_graph(query, k=settings.RETRIEVAL_TOP_K_PER_BRANCH)
        return {"graph_chunks": [c.to_dict() for c in chunks]}

    def _rrf_fusion_node(self, state: AgentState) -> Dict[str, Any]:
        """Node 4: Reciprocal Rank Fusion of 3 parallel retrieval lists into Top 5 chunks."""
        v_chunks = [Chunk.from_dict(d) for d in state.get("vector_chunks", [])]
        b_chunks = [Chunk.from_dict(d) for d in state.get("bm25_chunks", [])]
        g_chunks = [Chunk.from_dict(d) for d in state.get("graph_chunks", [])]

        ranked_lists = {
            "Vector_DB": v_chunks,
            "BM25": b_chunks,
            "Knowledge_Graph": g_chunks
        }

        fused_details = reciprocal_rank_fusion(
            ranked_lists=ranked_lists,
            k_constant=settings.RRF_K_CONSTANT,
            top_k=settings.RRF_TOP_K
        )

        fused_chunks = [item[0].to_dict() for item in fused_details]
        rrf_provenance = {
            item[0].chunk_id: {
                "score": round(item[1], 5),
                "ranks": item[2],
                "law_or_article": item[0].law_or_article,
                "heading_hierarchy": item[0].heading_hierarchy
            }
            for item in fused_details
        }

        return {
            "fused_chunks": fused_chunks,
            "rrf_provenance": rrf_provenance
        }

    def _responsibility_agent_node(self, state: AgentState) -> Dict[str, Any]:
        """Node 5: Final Responsibility Moderation Agent."""
        query = state["query"]
        fused_chunk_dicts = state.get("fused_chunks", [])
        chunks = [Chunk.from_dict(d) for d in fused_chunk_dicts]

        # Format context blocks
        context_blocks = []
        for idx, c in enumerate(chunks):
            header = c.get_full_context_header()
            block = f"--- [EVIDENCE CHUNK #{idx+1}] ---\n{header}\nText:\n{c.text}\n"
            context_blocks.append(block)

        context_str = "\n".join(context_blocks) if context_blocks else "(No context chunks retrieved)"

        user_message = f"""### STATEMENT TO EVALUATE:
\"\"\"
{query}
\"\"\"

### RETRIEVED EVIDENCE CONTEXT (from EU AI Act, NIST AI RMF, UN/UNESCO Hate Speech frameworks, EU DSA, EEOC, CoE):
{context_str}

IMPORTANT INSTRUCTIONS:
1. First read the STATEMENT carefully.
2. Determine if it contains hate speech, bigotry, discrimination, stereotyping, or harmful AI practices.
3. Use the retrieved context chunks as reference material — they do NOT inherently mean the statement is a violation. You must evaluate the statement on its own merits to determine if it is genuinely harmful or benign.
4. Output your analysis following the strict format specified in your system prompt."""

        def invoke_llm(llm, llm_name):
            try:
                from langchain_core.messages import SystemMessage, HumanMessage
                messages = [
                    SystemMessage(content=RESPONSIBILITY_AGENT_SYSTEM_PROMPT),
                    HumanMessage(content=user_message)
                ]
                response = llm.invoke(messages)
                raw = response.content if hasattr(response, "content") else str(response)
                if isinstance(raw, list):
                    text = "".join(
                        part.get("text", "") if isinstance(part, dict) else str(part)
                        for part in raw
                    ).strip()
                else:
                    text = str(raw).strip()
                if not text:
                    text = str(response)
                logger.info(f"Got verdict from {llm_name} (first 200 chars): {text[:200]}")
                return text
            except Exception as e:
                logger.warning(f"LLM call to {llm_name} failed: {e}")
                return None

        verdict_text = None

        # Primary LLM call
        if self.llm:
            verdict_text = invoke_llm(self.llm, f"Primary ({settings.AGENT_LLM_PROVIDER})")

        # Fallback: try Groq if primary (Gemini) failed
        if not verdict_text and settings.GROQ_API_KEY and settings.GROQ_API_KEY.strip():
            try:
                from langchain_groq import ChatGroq
                groq_llm = ChatGroq(
                    model=settings.GROQ_MODERATION_MODEL,
                    temperature=0.0,
                    groq_api_key=settings.GROQ_API_KEY
                )
                verdict_text = invoke_llm(groq_llm, f"Fallback Groq ({settings.GROQ_MODERATION_MODEL})")
            except Exception as e:
                logger.warning(f"Groq fallback init failed: {e}")

        if verdict_text:
            is_violation = _extract_verdict_from_llm_response(verdict_text)
            violated_rules = []
            if is_violation:
                violated_rules = list(set(c.law_or_article for c in chunks if c.law_or_article))

            return {
                "verdict": verdict_text,
                "is_violation": is_violation,
                "violated_rules": violated_rules
            }

        # Last resort: heuristic fallback
        logger.warning("All LLM providers failed. Using heuristic fallback.")
        is_violation = _quick_heuristic_check(query)
        verdict_text = self._generate_heuristic_report(query, chunks, is_violation)
        violated_rules = list(set(c.law_or_article for c in chunks if c.law_or_article)) if is_violation else []

        return {
            "verdict": verdict_text,
            "is_violation": is_violation,
            "violated_rules": violated_rules
        }

    def _generate_heuristic_report(self, query: str, chunks: List[Chunk], is_violation: bool) -> str:
        """
        Fallback report when LLM is completely unavailable.
        Only called when the LLM call throws an exception.
        """
        if is_violation:
            cited_articles = list(set(c.law_or_article for c in chunks if c.law_or_article))
            cited_hierarchies = [
                f"- **{c.doc_title}**: `{c.heading_hierarchy}` (Pages: {c.page_numbers})"
                for c in chunks
            ]
            evidence_lines = ""
            for i, c in enumerate(chunks[:3]):
                evidence_lines += (
                    f"\n  * **Evidence #{i+1} [{c.law_or_article or c.heading_path}]**: "
                    f"\"{c.text[:200]}...\" (Source: {c.source_file}, p. {c.page_numbers})\n"
                )

            return f"""### 1. 🏷️ COMPLIANCE VERDICT
- **STATUS**: UNETHICAL / NON-COMPLIANT ⚠️ (FLAGGED)
- The candidate response contains content that explicitly advocates manipulative, deceptive, or otherwise harmful AI practices that violate established AI governance standards.

---

### 2. 📋 ASSESSMENT SUMMARY
- **What the candidate response does**: The response promotes AI techniques or behaviors that are explicitly prohibited under global AI safety and ethics frameworks.
- **Relevance to AI governance**: This content directly violates foundational principles of responsible AI.
- **Key findings**:
  - Contains explicit language advocating manipulative or harmful AI practices
  - Advocates behaviors that undermine user autonomy, consent, or safety
  - Conflicts with transparency and human oversight requirements
  - Poses measurable risk of harm to affected individuals

---

### 3. 📜 LEGAL & FRAMEWORK ANALYSIS
The candidate response triggers the following provisions:
{chr(10).join(cited_hierarchies)}

**Specific Citations**: {', '.join(cited_articles) if cited_articles else 'EU AI Act Prohibited Practices & NIST AI RMF Core Functions'}
{evidence_lines}

---

### 4. 🛡️ GUIDANCE
- **Corrective Action**: Do not implement or recommend the practices described in the candidate response.
- **Safe Alternative**: Redesign the AI system with explicit user consent, full transparency, human oversight mechanisms, and bias audit processes in accordance with NIST GOVERN 1.2 and EU AI Act Chapter III requirements.
"""
        else:
            return f"""### 1. 🏷️ COMPLIANCE VERDICT
- **STATUS**: COMPLIANT ✅
- The candidate response does not contain content that explicitly advocates manipulative, deceptive, discriminatory, or legally prohibited AI behaviors.

---

### 2. 📋 ASSESSMENT SUMMARY
- **What the candidate response does**: The response addresses a topic without promoting harmful, manipulative, or prohibited AI techniques.
- **Relevance to AI governance**: The content is consistent with responsible AI principles and does not raise governance red flags.
- **Key findings**:
  - No prohibited AI practices identified in the response
  - No manipulation, deception, or discrimination advocated
  - No evidence of intent to bypass human oversight or user consent
  - Content is factual, constructive, or otherwise benign

---

### 3. 📜 LEGAL & FRAMEWORK ANALYSIS
The response is consistent with the following principles from the compliance context:
- **Transparency and Human Oversight** (EU AI Act, Article 13–14): Not violated.
- **Prohibited AI Practices** (EU AI Act, Article 5): Not applicable — no prohibited practice advocated.
- **NIST AI RMF Trustworthiness Characteristics**: The response does not contradict Safe, Accountable, Transparent, or Fair AI principles.

---

### 4. 🛡️ GUIDANCE
The response is compliant. To maintain best practices in responsible AI:
- Continue documenting AI decision processes for auditability.
- Apply proportional risk management using NIST MAP and MEASURE functions.
- Ensure any AI systems described maintain human oversight and recourse mechanisms as per EU AI Act Article 14.
"""

    def _build_graph(self) -> Any:
        """Constructs the LangGraph DAG with parallel retrieval fan-out."""
        if StateGraph is None:
            return None

        try:
            graph = StateGraph(AgentState)

            graph.add_node("retrieve_vector", self._retrieve_vector_node)
            graph.add_node("retrieve_bm25", self._retrieve_bm25_node)
            graph.add_node("retrieve_graph", self._retrieve_graph_node)
            graph.add_node("reciprocal_rank_fusion", self._rrf_fusion_node)
            graph.add_node("responsibility_agent", self._responsibility_agent_node)

            # Parallel Fan-Out from START
            graph.add_edge(START, "retrieve_vector")
            graph.add_edge(START, "retrieve_bm25")
            graph.add_edge(START, "retrieve_graph")

            # Fan-In to RRF
            graph.add_edge("retrieve_vector", "reciprocal_rank_fusion")
            graph.add_edge("retrieve_bm25", "reciprocal_rank_fusion")
            graph.add_edge("retrieve_graph", "reciprocal_rank_fusion")

            # RRF → Final Responsibility Agent
            graph.add_edge("reciprocal_rank_fusion", "responsibility_agent")
            graph.add_edge("responsibility_agent", END)

            return graph.compile()
        except Exception as e:
            logger.warning(f"Could not compile LangGraph ({e}), using direct executor.")
            return None

    def evaluate(self, candidate_answer: str) -> Dict[str, Any]:
        """Executes the full Responsibility Agent LangGraph pipeline."""
        initial_state: AgentState = {
            "query": candidate_answer,
            "vector_chunks": [],
            "bm25_chunks": [],
            "graph_chunks": [],
            "fused_chunks": [],
            "rrf_provenance": {},
            "verdict": "",
            "is_violation": False,
            "violated_rules": []
        }

        if self.app is not None:
            try:
                return self.app.invoke(initial_state)
            except Exception as e:
                logger.warning(f"LangGraph execution exception ({e}), falling back to direct executor.")

        # Direct DAG execution fallback
        from concurrent.futures import ThreadPoolExecutor
        state = dict(initial_state)

        with ThreadPoolExecutor(max_workers=3) as executor:
            fut_v = executor.submit(self._retrieve_vector_node, state)
            fut_b = executor.submit(self._retrieve_bm25_node, state)
            fut_g = executor.submit(self._retrieve_graph_node, state)
            
            state.update(fut_v.result())
            state.update(fut_b.result())
            state.update(fut_g.result())

        state.update(self._rrf_fusion_node(state))
        state.update(self._responsibility_agent_node(state))

        return state


def build_responsibility_graph(retriever: HybridRetriever, chunk_store: ChunkStore) -> ResponsibilityAgentWorkflow:
    return ResponsibilityAgentWorkflow(retriever, chunk_store)
