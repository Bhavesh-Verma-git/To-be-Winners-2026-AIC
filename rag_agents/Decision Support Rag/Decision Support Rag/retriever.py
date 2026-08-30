import os
import json
import pickle
import time
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

DATA_DIR = os.path.join(os.path.dirname(__file__), "Data")

class HybridRetriever:
    def __init__(self, data_dir=DATA_DIR, embedding_model_name="all-MiniLM-L6-v2"):
        self.data_dir = data_dir
        self.embedding_model_name = embedding_model_name
        self.embed_model = None
        self.faiss_index = None
        self.bm25 = None
        self.parents_store = None
        self.children_store = None
        self.is_loaded = False
        
    def load(self):
        """Loads indices and parent-child stores into memory."""
        if self.is_loaded:
            return
            
        print("Loading Hybrid Retriever components...")
        t0 = time.time()
        
        # Load stores
        parents_file = os.path.join(self.data_dir, "parents_store.json")
        children_file = os.path.join(self.data_dir, "children_store.json")
        
        with open(parents_file, "r", encoding="utf-8") as f:
            self.parents_store = json.load(f)
            
        with open(children_file, "r", encoding="utf-8") as f:
            self.children_store = json.load(f)
            
        # Load FAISS index
        faiss_file = os.path.join(self.data_dir, "faiss_index.bin")
        self.faiss_index = faiss.read_index(faiss_file)
        
        # Load BM25 index
        bm25_file = os.path.join(self.data_dir, "bm25_index.pkl")
        with open(bm25_file, "rb") as f:
            bm25_data = pickle.load(f)
            self.bm25 = bm25_data["bm25"]
            
        # Load embedding model
        self.embed_model = SentenceTransformer(self.embedding_model_name)
        
        self.is_loaded = True
        print(f"Hybrid Retriever loaded in {time.time() - t0:.2f}s")
        
    def search(self, query: str, top_k: int = 5, dense_weight: float = 0.5, sparse_weight: float = 0.5):
        """
        Performs hybrid retrieval (Dense FAISS + Sparse BM25) and resolves to unique parent chunks.
        
        Returns:
            dict containing:
                - 'results': list of retrieved parent chunks with matched child metadata & scores
                - 'context_text': clean combined dialogue chunk context for LLM
                - 'latency_ms': retrieval latency in milliseconds
        """
        if not self.is_loaded:
            self.load()
            
        start_time = time.time()
        
        # 1. Dense Vector Search
        q_emb = self.embed_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        
        # Search candidate pool (top 50 candidates from FAISS)
        candidate_k = min(len(self.children_store), max(top_k * 10, 50))
        dense_distances, dense_indices = self.faiss_index.search(q_emb, candidate_k)
        
        dense_scores = {}
        for dist, idx in zip(dense_distances[0], dense_indices[0]):
            if idx >= 0:
                norm_score = max(0.0, float(dist))
                dense_scores[int(idx)] = norm_score
                
        # 2. Sparse BM25 Search with acronym expansion (e.g. LCD -> l_c_d_, TV -> t_v_)
        raw_tokens = query.lower().split()
        expanded_tokens = list(raw_tokens)
        for token in raw_tokens:
            clean_t = token.strip("?,.!;:\"'()")
            if len(clean_t) in [2, 3, 4] and clean_t.isalpha():
                # Form AMI corpus acronym format (e.g., 'lcd' -> 'l_c_d_')
                underscored = "_".join(list(clean_t)) + "_"
                if underscored not in expanded_tokens:
                    expanded_tokens.append(underscored)
            if "_" in clean_t:
                # Also expand from 'l_c_d_' -> 'lcd'
                de_underscored = clean_t.replace("_", "")
                if de_underscored and de_underscored not in expanded_tokens:
                    expanded_tokens.append(de_underscored)
                    
        bm25_raw_scores = np.array(self.bm25.get_scores(expanded_tokens))
        
        max_bm25 = float(np.max(bm25_raw_scores)) if len(bm25_raw_scores) > 0 else 0.0
        min_bm25 = float(np.min(bm25_raw_scores)) if len(bm25_raw_scores) > 0 else 0.0
        bm25_range = max_bm25 - min_bm25 if max_bm25 > min_bm25 else 1.0
        
        # Candidate set from top Dense + top BM25
        top_bm25_indices = np.argpartition(bm25_raw_scores, -candidate_k)[-candidate_k:]
        
        candidate_indices = set(dense_scores.keys()).union(set(top_bm25_indices.tolist()))
            
        combined_candidates = []
        for idx in candidate_indices:
            d_score = dense_scores.get(idx, 0.0)
            b_score = (bm25_raw_scores[idx] - min_bm25) / bm25_range if max_bm25 > 0 else 0.0
            
            hybrid_score = (dense_weight * d_score) + (sparse_weight * b_score)
            
            child_info = self.children_store[idx]
            combined_candidates.append({
                "child_idx": idx,
                "child_id": child_info["child_id"],
                "parent_id": child_info["parent_id"],
                "meeting_id": child_info["meeting_id"],
                "speaker": child_info["speaker"],
                "child_text": child_info["text"],
                "hybrid_score": float(hybrid_score),
                "dense_score": float(d_score),
                "bm25_score": float(b_score)
            })
            
        # Sort by hybrid score descending
        combined_candidates.sort(key=lambda x: x["hybrid_score"], reverse=True)
        
        # 4. Resolve to unique Parent Chunks (preserving highest child score)
        seen_parents = set()
        resolved_results = []
        
        for cand in combined_candidates:
            p_id = cand["parent_id"]
            if p_id not in seen_parents:
                seen_parents.add(p_id)
                parent_obj = self.parents_store.get(p_id)
                if parent_obj:
                    resolved_results.append({
                        "parent_id": p_id,
                        "meeting_id": parent_obj["meeting_id"],
                        "speaker": parent_obj["speaker"],
                        "parent_text": parent_obj["text"], # Strictly chunk data for context
                        "metadata": parent_obj["metadata"], # Contains meeting summary and split
                        "matched_child": cand["child_text"],
                        "child_id": cand["child_id"],
                        "hybrid_score": cand["hybrid_score"],
                        "dense_score": cand["dense_score"],
                        "bm25_score": cand["bm25_score"]
                    })
            if len(resolved_results) >= top_k:
                break
                
        # 5. Format pure chunk data context for LLM prompt
        context_chunks = []
        for i, res in enumerate(resolved_results):
            context_chunks.append(
                f"[Source Chunk {i+1} | Meeting ID: {res['meeting_id']} | Speaker: {res['speaker']}]\n"
                f"{res['parent_text']}"
            )
            
        context_text = "\n\n".join(context_chunks)
        latency_ms = (time.time() - start_time) * 1000
        
        return {
            "results": resolved_results,
            "context_text": context_text,
            "latency_ms": round(latency_ms, 2)
        }
