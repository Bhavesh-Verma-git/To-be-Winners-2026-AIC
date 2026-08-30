from typing import List, Dict, Tuple
from ..ingestion.chunk_store import Chunk

def reciprocal_rank_fusion(
    ranked_lists: Dict[str, List[Chunk]],
    k_constant: int = 60,
    top_k: int = 5
) -> List[Tuple[Chunk, float, Dict[str, int]]]:
    """
    Executes Reciprocal Rank Fusion (RRF) across multiple retrieved ranked lists.
    
    Formula:
        RRF_Score(d) = sum_{r in ranked_lists} (1.0 / (k_constant + rank(d, r)))

    Parameters:
        ranked_lists: Dict mapping retriever_name -> List of Chunk objects
        k_constant: Smoothing parameter (standard is 60)
        top_k: Number of top ranked chunks to return (default 5)

    Returns:
        List of tuples: (Chunk, rrf_score, {retriever_name: 1-based rank})
    """
    rrf_scores: Dict[str, float] = {}
    chunk_map: Dict[str, Chunk] = {}
    rank_provenance: Dict[str, Dict[str, int]] = {}

    for retriever_name, chunk_list in ranked_lists.items():
        for rank_idx, chunk in enumerate(chunk_list):
            rank_1based = rank_idx + 1
            cid = chunk.chunk_id
            chunk_map[cid] = chunk

            # Compute score contribution
            score_contrib = 1.0 / (k_constant + rank_1based)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + score_contrib

            if cid not in rank_provenance:
                rank_provenance[cid] = {}
            rank_provenance[cid][retriever_name] = rank_1based

    # Sort chunks by RRF score descending
    sorted_chunk_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

    results: List[Tuple[Chunk, float, Dict[str, int]]] = []
    for cid in sorted_chunk_ids[:top_k]:
        chunk = chunk_map[cid]
        score = rrf_scores[cid]
        provenance = rank_provenance[cid]
        results.append((chunk, score, provenance))

    return results
