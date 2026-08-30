#!/usr/bin/env python3
"""
rag_agent.py
Hybrid Toxic RAG Agent.

Features:
- Hybrid Retrieval: Dense Vector Search (FAISS, 70% weight) + Sparse BM25 (30% weight).
- Retrieves exactly top 7 chunks.
- Row-level chunks: 1 row = 1 full document chunk with rich safety metadata.
- Fast, one-time persistent disk indexing for instant loading during inference.
- Fast LLM inference with ChatGroq.
"""

import os
import re
import time
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional

# Disable tokenizers fork warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_groq import ChatGroq

# Load environment variables from .env
load_dotenv(override=True)

DEFAULT_DATASET_PATH = "Dataset/final_tox_Rag.csv"
DEFAULT_CACHE_DIR = ".index_cache"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
DEFAULT_TOP_K = 7
VECTOR_WEIGHT = 0.5
BM25_WEIGHT = 0.5


def clean_llm_response(text: str, docs: Optional[List[Document]] = None) -> str:
    """
    Cleans raw LLM response by stripping thinking tags, chain-of-thought blocks,
    and meta-commentary, leaving only the final formulated answer.
    """
    if not text:
        if docs:
            return "\n".join([f"- {d.page_content}" for d in docs])
        return ""

    # Check for Draft / Final Answer section in reasoning output
    draft_match = re.search(r"(?:\*Draft:\*|Draft:|\*Final Answer:\*|Final Answer:)\s*\n(.*)", text, flags=re.DOTALL)
    if draft_match:
        text = draft_match.group(1).strip()
        text = re.sub(r"\n\s*(?:-?\s*Wait|-?\s*Let\'s|-?\s*Actually|-?\s*I will|-?\s*The prompt|This directly answers|All constraints met|Output matches|Check against|Note:|Constraints).*$", "", text, flags=re.DOTALL).strip()
        return text

    # Strip XML thinking tags (complete and partial)
    outside = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    outside = re.sub(r"<think>.*", "", outside, flags=re.DOTALL).strip()

    if outside and not outside.startswith("Here's a thinking process") and len(outside) > 10:
        cleaned = re.sub(r"</?think>", "", outside).strip()
        cleaned = re.sub(r"\n\s*(?:-?\s*Wait|-?\s*Let\'s|-?\s*Actually|-?\s*I will|-?\s*The prompt|This directly answers|All constraints met|Output matches|Check against|Note:|Constraints).*$", "", cleaned, flags=re.DOTALL).strip()
        return cleaned

    # If answer was embedded inside a think/draft block
    match = re.search(
        r"(?:Based on|Here are|Here is|The following|According to|In the provided|From the retrieved)[^\n]*\n(?:[ \t]*[-*•\d].*\n?)+",
        text,
        flags=re.DOTALL
    )
    if match:
        extracted = match.group(0).strip()
        extracted = re.sub(r"\n\s*(?:-?\s*Wait|-?\s*Let\'s|-?\s*Actually|-?\s*I will|-?\s*The prompt|This directly answers|All constraints met|Output matches|Check against|Note:|Constraints).*", "", extracted, flags=re.DOTALL)
        return extracted.strip()

    # Search for bullet lists of chunks
    bullets = re.findall(r"(?:^[ \t]*[-*•\d]+\.?\s+[\"\'\w].*\n?)+", text, flags=re.MULTILINE)
    if bullets:
        res_bullets = bullets[-1].strip()
        res_bullets = re.sub(r"\n\s*(?:-?\s*Wait|-?\s*Let\'s|-?\s*Actually|-?\s*I will|-?\s*The prompt|This directly answers|All constraints met|Output matches|Check against|Note:|Constraints).*", "", res_bullets, flags=re.DOTALL)
        return res_bullets.strip()

    # Fallback to direct formulation from retrieved docs if LLM output was completely stripped
    if docs:
        return "\n".join([f"- \"{d.page_content}\"" for d in docs])

    return re.sub(r"</?think>", "", text).strip()


def load_dataset_as_documents(csv_path: str = DEFAULT_DATASET_PATH) -> List[Document]:
    """
    Loads final_tox_Rag.csv into LangChain Document objects.
    Each row's 'Text' is the full chunk (no sub-chunking).
    All other columns are stored as metadata.
    """
    path = Path(csv_path)
    if not path.exists():
        alt_path = Path("dataset") / path.name
        if alt_path.exists():
            path = alt_path
        else:
            raise FileNotFoundError(f"Dataset not found at {csv_path} or {alt_path}. Please run prepare_dataset.py first.")

    df = pd.read_csv(path)
    df = df.fillna("")

    documents: List[Document] = []
    for idx, row in df.iterrows():
        text = str(row.get("Text", "")).strip()
        if not text:
            continue

        metadata = {
            "row_id": int(idx),
            "target group": str(row.get("target group", "")).strip(),
            "factual": str(row.get("factual", "")).strip(),
            "in-group effect": str(row.get("in-group effect", "")).strip(),
            "framing": str(row.get("framing", "")).strip(),
            "lewd": str(row.get("lewd", "")).strip(),
            "predicted group": str(row.get("predicted group", "")).strip(),
            "stereotyping": str(row.get("stereotyping", "")).strip(),
        }

        documents.append(Document(page_content=text, metadata=metadata))

    return documents


class HybridEnsembleRetriever(BaseRetriever):
    """
    Hybrid retriever combining Dense Vector retriever (FAISS, 70%) and Sparse BM25 retriever (30%)
    using Weighted Reciprocal Rank Fusion (RRF) to retrieve the top_k chunks.
    """
    vector_retriever: Any
    bm25_retriever: Any
    vector_weight: float = 0.7
    bm25_weight: float = 0.3
    top_k: int = 7
    c: int = 60

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        candidate_k = max(self.top_k * 2, 20)
        
        # 1. Vector Search (70% weight)
        try:
            vector_docs = self.vector_retriever.invoke(query)[:candidate_k]
        except Exception:
            vector_docs = self.vector_retriever.get_relevant_documents(query)[:candidate_k]

        # 2. BM25 Search (30% weight)
        try:
            bm25_docs = self.bm25_retriever.invoke(query)[:candidate_k]
        except Exception:
            bm25_docs = self.bm25_retriever.get_relevant_documents(query)[:candidate_k]

        # 3. Weighted Reciprocal Rank Fusion (RRF)
        doc_scores: Dict[str, Dict[str, Any]] = {}

        for rank, doc in enumerate(vector_docs):
            doc_id = str(doc.metadata.get("row_id", doc.page_content))
            score = self.vector_weight * (1.0 / (self.c + rank + 1))
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {"doc": doc, "score": score, "sources": ["vector"]}
            else:
                doc_scores[doc_id]["score"] += score
                doc_scores[doc_id]["sources"].append("vector")

        for rank, doc in enumerate(bm25_docs):
            doc_id = str(doc.metadata.get("row_id", doc.page_content))
            score = self.bm25_weight * (1.0 / (self.c + rank + 1))
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {"doc": doc, "score": score, "sources": ["bm25"]}
            else:
                doc_scores[doc_id]["score"] += score
                if "bm25" not in doc_scores[doc_id]["sources"]:
                    doc_scores[doc_id]["sources"].append("bm25")

        sorted_items = sorted(doc_scores.values(), key=lambda x: x["score"], reverse=True)
        top_results: List[Document] = []
        for item in sorted_items[:self.top_k]:
            doc = item["doc"]
            doc.metadata["retrieval_score"] = round(item["score"], 6)
            doc.metadata["matched_retrievers"] = item["sources"]
            top_results.append(doc)

        return top_results


class ToxicRAGAgent:
    """
    End-to-End Toxic RAG Agent with pre-indexed database loading.
    """

    def __init__(
        self,
        dataset_path: str = DEFAULT_DATASET_PATH,
        cache_dir: str = DEFAULT_CACHE_DIR,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        groq_model: str = DEFAULT_GROQ_MODEL,
        top_k: int = DEFAULT_TOP_K,
        vector_weight: float = VECTOR_WEIGHT,
        bm25_weight: float = BM25_WEIGHT,
        force_rebuild_index: bool = False
    ):
        self.dataset_path = dataset_path
        self.cache_dir = Path(cache_dir)
        self.embedding_model_name = embedding_model_name
        self.groq_model = groq_model
        self.top_k = top_k
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.force_rebuild = force_rebuild_index

        self.embeddings: Optional[HuggingFaceEmbeddings] = None
        self.vector_store: Optional[FAISS] = None
        self.bm25_retriever: Optional[BM25Retriever] = None
        self.hybrid_retriever: Optional[HybridEnsembleRetriever] = None
        self.llm: Optional[ChatGroq] = None
        self.rag_chain = None

        self._initialize_pipeline()

    def _initialize_pipeline(self):
        """Loads pre-built indices directly from disk for low-latency inference."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        faiss_index_dir = self.cache_dir / "faiss_index"
        bm25_cache_file = self.cache_dir / "bm25_retriever.pkl"

        # Initialize Embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

        if (
            not self.force_rebuild
            and faiss_index_dir.exists()
            and bm25_cache_file.exists()
        ):
            self.vector_store = FAISS.load_local(
                folder_path=str(faiss_index_dir),
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True
            )
            with open(bm25_cache_file, "rb") as f:
                self.bm25_retriever = pickle.load(f)
        else:
            docs = load_dataset_as_documents(self.dataset_path)
            self.vector_store = FAISS.from_documents(docs, self.embeddings)
            self.vector_store.save_local(str(faiss_index_dir))

            self.bm25_retriever = BM25Retriever.from_documents(docs)
            with open(bm25_cache_file, "wb") as f:
                pickle.dump(self.bm25_retriever, f)

        self.vector_retriever = self.vector_store.as_retriever(search_kwargs={"k": self.top_k})
        self.bm25_retriever.k = self.top_k

        # Hybrid Retriever: 70% FAISS (Dense) + 30% BM25 (Sparse)
        self.hybrid_retriever = HybridEnsembleRetriever(
            vector_retriever=self.vector_retriever,
            bm25_retriever=self.bm25_retriever,
            vector_weight=self.vector_weight,
            bm25_weight=self.bm25_weight,
            top_k=self.top_k
        )

        groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
        if groq_api_key:
            self._init_llm(groq_api_key)

    def _init_llm(self, api_key: str):
        """Initializes Groq LLM with fallback support."""
        models_to_try = [self.groq_model, "openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.6-27b"]
        for m in models_to_try:
            try:
                self.llm = ChatGroq(
                    groq_api_key=api_key,
                    model_name=m,
                    temperature=0.1,
                    max_tokens=1024,
                )
                self._build_rag_chain()
                self.groq_model = m
                break
            except Exception:
                continue

    def _build_rag_chain(self):
        """Prompt instructing the LLM to formulate what the user requested directly from the chunks."""
        prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        """Answer the user query using only the retrieved context.
Formulate a direct, natural answer from the relevant chunks. Combine information from multiple chunks when useful.
Do not add outside knowledge, assumptions, or unsupported information.
Do not summarize the retrieval process or describe the context; answer the user directly.
Output only the final answer. Never output reasoning, analysis, steps, or instructions.
If the context is irrelevant or insufficient, say:
"The retrieved context does not contain enough information to answer this query." """
    ),
    (
        "human",
        """Context:
{context}

User:
{query}

Answer:"""
    )
])

        self.rag_chain = prompt_template | self.llm | StrOutputParser()

    def format_context_docs(self, docs: List[Document]) -> str:
        """Formats retrieved chunks with their content and metadata for LLM prompt context."""
        formatted_chunks = []
        for i, doc in enumerate(docs, 1):
            meta = doc.metadata
            chunk_str = (
                f"[{i}] \"{doc.page_content}\" (Target: {meta.get('target group', '')}, "
                f"Factual: {meta.get('factual', '')}, Framing: {meta.get('framing', '')}, "
                f"Stereotyping: {meta.get('stereotyping', '')})"
            )
            formatted_chunks.append(chunk_str)
        return "\n".join(formatted_chunks)

    def query(self, query_text: str) -> Dict[str, Any]:
        """
        Executes Hybrid RAG query:
        1. 70/30 Hybrid retrieval (0.7 FAISS + 0.3 BM25, 7 chunks).
        2. Direct formulation and answer generation from retrieved chunks.
        """
        overall_start = time.time()

        # Step 1: Hybrid Retrieval directly from pre-loaded database
        retrieval_start = time.time()
        docs = self.hybrid_retriever.invoke(query_text)
        retrieval_latency = time.time() - retrieval_start

        formatted_context = self.format_context_docs(docs)

        # Step 2: Generation via Groq
        if self.rag_chain is None:
            groq_key = os.getenv("GROQ_API_KEY", "").strip()
            if groq_key:
                self._init_llm(groq_key)

        answer = ""
        generation_latency = 0.0

        if self.rag_chain is not None:
            gen_start = time.time()
            raw_answer = self.rag_chain.invoke({
                "context": formatted_context,
                "query": query_text
            })
            answer = clean_llm_response(raw_answer, docs=docs)
            generation_latency = time.time() - gen_start
        else:
            answer = "[GROQ_API_KEY NOT SET] Please configure GROQ_API_KEY in .env."

        total_latency = time.time() - overall_start

        return {
            "query": query_text,
            "answer": answer,
            "retrieved_chunks": docs,
            "formatted_context": formatted_context,
            "retrieval_latency_ms": round(retrieval_latency * 1000, 2),
            "generation_latency_ms": round(generation_latency * 1000, 2),
            "total_latency_ms": round(total_latency * 1000, 2),
        }


if __name__ == "__main__":
    agent = ToxicRAGAgent()
    res = agent.query("joke on asian")
    print("\nAnswer:\n", res["answer"])
    print(f"\nLatency: {res['total_latency_ms']} ms")
