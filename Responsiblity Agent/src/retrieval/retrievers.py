from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor

from ..ingestion.chunk_store import Chunk, ChunkStore
from ..storage.vector_store import VectorStoreManager
from ..storage.bm25_store import BM25StoreManager
from ..storage.graph_store import KnowledgeGraphManager
from .rrf import reciprocal_rank_fusion
from ..config import settings

class HybridRetriever:
    """Coordinates parallel 3-way retrieval across Vector DB, BM25, and Neo4j Knowledge Graph."""

    def __init__(
        self,
        chunk_store: ChunkStore,
        vector_store: VectorStoreManager,
        bm25_store: BM25StoreManager,
        graph_store: KnowledgeGraphManager,
        top_k_per_branch: int = settings.RETRIEVAL_TOP_K_PER_BRANCH,
        rrf_top_k: int = settings.RRF_TOP_K,
        rrf_k_constant: int = settings.RRF_K_CONSTANT
    ):
        self.chunk_store = chunk_store
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.graph_store = graph_store
        self.top_k_per_branch = top_k_per_branch
        self.rrf_top_k = rrf_top_k
        self.rrf_k_constant = rrf_k_constant

    def retrieve_vector(self, query: str, k: Optional[int] = None) -> List[Chunk]:
        """Retrieves top-k chunks from ChromaDB Vector Store."""
        limit = k or self.top_k_per_branch
        return self.vector_store.similarity_search(query, k=limit, chunk_store=self.chunk_store)

    def retrieve_bm25(self, query: str, k: Optional[int] = None) -> List[Chunk]:
        """Retrieves top-k chunks from BM25 Lexical Store."""
        limit = k or self.top_k_per_branch
        return self.bm25_store.search(query, k=limit, chunk_store=self.chunk_store)

    def retrieve_graph(self, query: str, k: Optional[int] = None) -> List[Chunk]:
        """Retrieves top-k chunks from Neo4j Knowledge Graph."""
        limit = k or self.top_k_per_branch
        return self.graph_store.query_graph_for_chunks(query, k=limit, chunk_store=self.chunk_store)

    def retrieve_parallel(
        self,
        query: str,
        k_per_branch: Optional[int] = None
    ) -> Dict[str, List[Chunk]]:
        """Runs the three retrievers in parallel threads for low latency."""
        limit = k_per_branch or self.top_k_per_branch
        results: Dict[str, List[Chunk]] = {}

        with ThreadPoolExecutor(max_workers=3) as executor:
            future_vector = executor.submit(self.retrieve_vector, query, limit)
            future_bm25 = executor.submit(self.retrieve_bm25, query, limit)
            future_graph = executor.submit(self.retrieve_graph, query, limit)

            results["vector"] = future_vector.result()
            results["bm25"] = future_bm25.result()
            results["graph"] = future_graph.result()

        return results

    def retrieve_and_fuse(
        self,
        query: str,
        k_per_branch: Optional[int] = None,
        top_k_fused: Optional[int] = None
    ) -> Tuple[List[Chunk], Dict[str, List[Chunk]], List[Tuple[Chunk, float, Dict[str, int]]]]:
        """
        Executes parallel retrieval across Vector, BM25, and Graph, then fuses with RRF.
        Returns:
            (top_5_chunks, raw_branch_results, rrf_detailed_results)
        """
        branch_results = self.retrieve_parallel(query, k_per_branch)
        fused_details = reciprocal_rank_fusion(
            ranked_lists=branch_results,
            k_constant=self.rrf_k_constant,
            top_k=top_k_fused or self.rrf_top_k
        )
        top_chunks = [item[0] for item in fused_details]
        return top_chunks, branch_results, fused_details
