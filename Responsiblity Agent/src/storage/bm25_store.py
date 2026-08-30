import re
import math
import pickle
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from collections import Counter

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    class BM25Okapi:  # type: ignore
        """Pure Python fallback implementation of BM25Okapi."""
        def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25):
            self.k1 = k1
            self.b = b
            self.epsilon = epsilon
            self.corpus_size = len(corpus)
            self.avgdl = sum(len(doc) for doc in corpus) / max(self.corpus_size, 1)
            self.doc_freqs: List[Dict[str, int]] = []
            self.idf: Dict[str, float] = {}
            self.doc_len: List[int] = []
            
            nd: Dict[str, int] = {}
            for document in corpus:
                self.doc_len.append(len(document))
                frequencies = Counter(document)
                self.doc_freqs.append(frequencies)
                for word in frequencies.keys():
                    nd[word] = nd.get(word, 0) + 1

            idf_sum = 0
            negative_idfs = []
            for word, freq in nd.items():
                idf_val = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
                self.idf[word] = idf_val
                idf_sum += idf_val
                if idf_val < 0:
                    negative_idfs.append(word)

            average_idf = idf_sum / max(len(self.idf), 1)
            eps = self.epsilon * average_idf
            for word in negative_idfs:
                self.idf[word] = eps

        def get_scores(self, query: List[str]) -> List[float]:
            score = [0.0] * self.corpus_size
            doc_len = self.doc_len
            for q in query:
                q_freq = [doc.get(q, 0) for doc in self.doc_freqs]
                idf = self.idf.get(q, 0.0)
                for i in range(self.corpus_size):
                    denom = q_freq[i] + self.k1 * (1 - self.b + self.b * doc_len[i] / max(self.avgdl, 1e-6))
                    if denom > 0:
                        score[i] += (idf * q_freq[i] * (self.k1 + 1)) / denom
            return score

from ..ingestion.chunk_store import Chunk, ChunkStore
from ..config import settings

def simple_tokenize(text: str) -> List[str]:
    """Tokenize and normalize text for BM25 indexing."""
    cleaned = re.sub(r"[^\w\s-]", " ", text.lower())
    tokens = [t.strip() for t in cleaned.split() if len(t.strip()) > 1]
    return tokens


class BM25StoreManager:
    """Manages the BM25 lexical search index over compliance chunks."""

    def __init__(self, index_path: Path | str = settings.BM25_STORE_PATH):
        self.index_path = Path(index_path)
        self.bm25: Optional[BM25Okapi] = None
        self.chunk_ids: List[str] = []

    def build_index(self, chunks: List[Chunk]):
        """Builds BM25 index from a list of chunks."""
        self.chunk_ids = [c.chunk_id for c in chunks]
        corpus = [f"{c.heading_path} {c.law_or_article} {c.text}" for c in chunks]
        tokenized_corpus = [simple_tokenize(doc) for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def save_index(self, path: Optional[Path | str] = None):
        """Serializes BM25 index and chunk ID map to disk."""
        target_path = Path(path) if path else self.index_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "wb") as f:
            pickle.dump({
                "bm25": self.bm25,
                "chunk_ids": self.chunk_ids
            }, f)

    def load_index(self, path: Optional[Path | str] = None) -> bool:
        """Loads BM25 index from disk."""
        target_path = Path(path) if path else self.index_path
        if not target_path.exists():
            return False
        with open(target_path, "rb") as f:
            data = pickle.load(f)
            self.bm25 = data["bm25"]
            self.chunk_ids = data["chunk_ids"]
        return True

    def search(
        self,
        query: str,
        k: int = 4,
        chunk_store: Optional[ChunkStore] = None
    ) -> List[Chunk]:
        """Performs BM25 search and returns top-k Chunk objects."""
        if not self.bm25 or not self.chunk_ids:
            return []

        tokenized_query = set(simple_tokenize(query))
        if not tokenized_query:
            return []

        # Early exit: only score docs that share at least one token with the query
        valid_indices = []
        for i, doc_freq in enumerate(self.bm25.doc_freqs):
            if any(q in doc_freq for q in tokenized_query):
                valid_indices.append(i)
                
        if not valid_indices:
            return []

        # Convert back to list for get_scores compatibility
        tokenized_query_list = list(tokenized_query)
        doc_scores = self.bm25.get_scores(tokenized_query_list)
        
        # Rank document indices by score descending, but only valid ones
        valid_scores = [(i, doc_scores[i]) for i in valid_indices if doc_scores[i] > 0]
        valid_scores.sort(key=lambda x: x[1], reverse=True)
        top_k_indices = [item[0] for item in valid_scores[:k]]

        top_chunk_ids = [self.chunk_ids[i] for i in top_k_indices if doc_scores[i] > 0]

        # If zero-score matches, still pick top non-empty
        if not top_chunk_ids and ranked_indices:
            top_chunk_ids = [self.chunk_ids[i] for i in ranked_indices[:k]]

        if chunk_store:
            return chunk_store.get_chunks(top_chunk_ids)
        return []
