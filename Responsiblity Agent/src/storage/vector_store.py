import os
import math
import random
import hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path
from functools import lru_cache

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

try:
    import chromadb
except ImportError:
    chromadb = None  # type: ignore

from ..ingestion.chunk_store import Chunk, ChunkStore
from ..config import settings


def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Pure Python cosine similarity between two float vectors."""
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 > 0 and norm2 > 0:
        return dot / (norm1 * norm2)
    return 0.0


class FallbackEmbedder:
    """Fast deterministic hashing embedder for zero-dependency environments."""
    def __init__(self, dim: int = 384):
        self.dim = dim

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        seed = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)
        rng = random.Random(seed)
        vec = [rng.gauss(0, 1) for _ in range(self.dim)]
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec


def get_embedding_function():
    """Returns an embedding function based on available API keys or local fallback."""
    if settings.OPENAI_API_KEY and settings.OPENAI_API_KEY.strip():
        try:
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(
                openai_api_key=settings.OPENAI_API_KEY,
                model=settings.OPENAI_EMBEDDING_MODEL
            )
        except Exception:
            pass

    if settings.GOOGLE_API_KEY and settings.GOOGLE_API_KEY.strip().startswith("AIza"):
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            return GoogleGenerativeAIEmbeddings(
                google_api_key=settings.GOOGLE_API_KEY,
                model="models/text-embedding-004"
            )
        except Exception:
            pass

    return FallbackEmbedder(dim=384)


class VectorStoreManager:
    """Manages the ChromaDB / in-memory vector database for compliance chunks."""

    def __init__(
        self,
        persist_dir: Path | str = settings.VECTOR_DB_DIR,
        collection_name: str = "compliance_knowledge_base"
    ):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.embedder = get_embedding_function()
        self.client = None
        self.collection = None
        self._memory_chunks: List[Dict[str, Any]] = []

        if chromadb is not None:
            try:
                self.client = chromadb.PersistentClient(path=str(self.persist_dir))
                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"description": "EU AI Act and NIST AI RMF compliance chunks"}
                )
            except Exception:
                self.collection = None

    def add_chunks(self, chunks: List[Chunk], batch_size: int = 100):
        """Indexes chunks into ChromaDB and in-memory store."""
        if not chunks:
            return

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            ids = [c.chunk_id for c in batch]
            documents = [c.text for c in batch]
            
            # Sanitize metadata for ChromaDB
            metadatas: List[Dict[str, Any]] = []
            for c in batch:
                meta = {
                    "chunk_id": c.chunk_id,
                    "doc_title": c.doc_title,
                    "source_file": c.source_file,
                    "heading_path": c.heading_path,
                    "heading_hierarchy": c.heading_hierarchy,
                    "law_or_article": c.law_or_article,
                    "pages": ",".join(map(str, c.page_numbers)),
                    "content_type": c.content_type,
                }
                metadatas.append(meta)

            try:
                embeddings = self.embedder.embed_documents(documents)
            except Exception:
                fallback = FallbackEmbedder(dim=384)
                embeddings = fallback.embed_documents(documents)

            if self.collection is not None:
                try:
                    self.collection.upsert(
                        ids=ids,
                        documents=documents,
                        embeddings=embeddings,
                        metadatas=metadatas
                    )
                except Exception:
                    pass

            for idx, c in enumerate(batch):
                self._memory_chunks.append({
                    "chunk_id": c.chunk_id,
                    "embedding": embeddings[idx],
                    "chunk": c
                })

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        chunk_store: Optional[ChunkStore] = None
    ) -> List[Chunk]:
        """Performs vector similarity search and returns top-k Chunk objects."""
        @lru_cache(maxsize=256)
        def _cached_embed(q: str) -> List[float]:
            try:
                return self.embedder.embed_query(q)
            except Exception:
                fallback = FallbackEmbedder(dim=384)
                return fallback.embed_query(q)

        query_embedding = _cached_embed(query)

        if self.collection is not None and self.collection.count() > 0:
            try:
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=min(k, self.collection.count())
                )
                if results and "ids" in results and results["ids"]:
                    retrieved_ids = results["ids"][0]
                    if chunk_store:
                        return chunk_store.get_chunks(retrieved_ids)
            except Exception:
                pass

        # In-memory pure Python cosine similarity fallback
        if self._memory_chunks:
            scores = []
            for item in self._memory_chunks:
                sim = _cosine_similarity(query_embedding, item["embedding"])
                scores.append((sim, item["chunk"]))
            
            scores.sort(key=lambda x: x[0], reverse=True)
            return [item[1] for item in scores[:k]]

        return []
