import logging
from pathlib import Path
from typing import Optional, Dict, Any

from .config import settings, ensure_data_directories, NIST_PDF_PATH, EU_AI_ACT_PDF_PATH, CONTENT_MODERATION_PDFS
from .ingestion.chunk_store import ChunkStore
from .ingestion.chunker import HierarchicalChunker
from .storage.vector_store import VectorStoreManager
from .storage.bm25_store import BM25StoreManager
from .storage.graph_store import KnowledgeGraphManager
from .retrieval.retrievers import HybridRetriever
from .agent.graph import ResponsibilityAgentWorkflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ResponsibilityPipeline")


class ResponsibilityPipeline:
    """Master Pipeline orchestrating Ingestion, Storage, Hybrid Retrieval, and LangGraph Moderation."""

    def __init__(self):
        ensure_data_directories()
        self.chunk_store = ChunkStore()
        self.vector_store = VectorStoreManager()
        self.bm25_store = BM25StoreManager()
        self.graph_store = KnowledgeGraphManager()
        self.retriever: Optional[HybridRetriever] = None
        self.workflow: Optional[ResponsibilityAgentWorkflow] = None

    def ingest(self, force_rebuild: bool = False):
        """
        Parses NIST and EU AI Act PDFs, creates hierarchical chunks with heading tracking,
        and builds the Vector DB, BM25 index, and Neo4j Knowledge Graph.
        """
        logger.info("=== Starting Responsibility Knowledge Base Ingestion ===")

        if not force_rebuild and settings.CHUNK_STORE_PATH.exists():
            logger.info(f"Loading existing chunk store from {settings.CHUNK_STORE_PATH}...")
            self.chunk_store.load_from_json(settings.CHUNK_STORE_PATH)
            logger.info(f"Loaded {len(self.chunk_store)} chunks from disk.")
        else:
            logger.info("Parsing documents and generating hierarchical chunks...")
            chunker = HierarchicalChunker()
            chunks = chunker.ingest_documents(
                str(NIST_PDF_PATH),
                str(EU_AI_ACT_PDF_PATH),
                self.chunk_store
            )
            self.chunk_store.save_to_json(settings.CHUNK_STORE_PATH)
            logger.info(f"Generated and saved {len(chunks)} chunks to {settings.CHUNK_STORE_PATH}.")

            logger.info("Parsing additional content moderation documents...")
            new_chunks = chunker.ingest_additional_documents(CONTENT_MODERATION_PDFS, self.chunk_store)
            self.chunk_store.save_to_json(settings.CHUNK_STORE_PATH)
            logger.info(f"Generated and saved {len(new_chunks)} additional chunks.")

        chunks = self.chunk_store.all_chunks()

        # 1. Build Vector Store
        logger.info("Indexing chunks into ChromaDB Vector Store...")
        self.vector_store.add_chunks(chunks)

        # 2. Build BM25 Index
        logger.info("Building BM25 lexical search index...")
        self.bm25_store.build_index(chunks)
        self.bm25_store.save_index()

        # 3. Build Knowledge Graph (Neo4j / Graph Triples)
        logger.info("Building Knowledge Graph and extracting entity triples...")
        self.graph_store.build_graph_from_chunks(chunks, use_llm_transformer=True)

        logger.info("=== Ingestion and Indexing Completed Successfully ===")

    def load(self):
        """Loads indexed data structures and initializes the LangGraph workflow."""
        if not settings.CHUNK_STORE_PATH.exists():
            logger.info("No existing chunk store found. Running automated ingestion...")
            self.ingest()
        else:
            self.chunk_store.load_from_json(settings.CHUNK_STORE_PATH)
            self.bm25_store.load_index()

        self.retriever = HybridRetriever(
            chunk_store=self.chunk_store,
            vector_store=self.vector_store,
            bm25_store=self.bm25_store,
            graph_store=self.graph_store,
            top_k_per_branch=settings.RETRIEVAL_TOP_K_PER_BRANCH,
            rrf_top_k=settings.RRF_TOP_K,
            rrf_k_constant=settings.RRF_K_CONSTANT
        )

        self.workflow = ResponsibilityAgentWorkflow(
            retriever=self.retriever,
            chunk_store=self.chunk_store
        )
        logger.info("Responsibility Agent LangGraph Workflow ready.")

    def evaluate(self, candidate_answer: str) -> Dict[str, Any]:
        """Evaluates a candidate AI output for ethics, compliance, and legal violations."""
        if not self.workflow:
            self.load()
        return self.workflow.evaluate(candidate_answer)
