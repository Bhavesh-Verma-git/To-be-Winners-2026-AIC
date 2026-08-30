from typing import List, Dict, Any, Optional
try:
    from typing import TypedDict
except ImportError:
    try:
        from typing_extensions import TypedDict  # type: ignore
    except ImportError:
        TypedDict = dict  # type: ignore

class AgentState(TypedDict):
    """LangGraph state schema for the Responsibility Agent."""
    query: str                                # The toxic / unethical RAG output to evaluate
    vector_chunks: List[Dict[str, Any]]       # Top 4 chunks from Vector DB
    bm25_chunks: List[Dict[str, Any]]         # Top 4 chunks from BM25
    graph_chunks: List[Dict[str, Any]]        # Top 4 chunks from Neo4j Knowledge Graph
    fused_chunks: List[Dict[str, Any]]        # Top 5 fused chunks from RRF
    rrf_provenance: Dict[str, Any]            # Details of RRF scores and retriever sources
    verdict: str                              # Final compliance moderation answer
    is_violation: bool                        # Boolean flag indicating whether ethical/legal rules were violated
    violated_rules: List[str]                 # Specific law articles or NIST rules cited
