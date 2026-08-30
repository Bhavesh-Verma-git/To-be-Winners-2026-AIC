"""
============================================================
  Customer Support RAG Agent — Step 1: Build the Index
  Run this ONCE to download data and build FAISS + BM25
============================================================
"""

import os
import sys
import pickle
from pathlib import Path
from dotenv import load_dotenv

# Load API keys from .env file
load_dotenv(Path(__file__).parent.parent.parent / ".env")

from datasets import load_dataset
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever

# ── Paths ────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
FAISS_INDEX_DIR = BASE_DIR / "faiss_index"
BM25_INDEX_PATH = BASE_DIR / "faiss_index" / "bm25_index.pkl"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # Free, local, fast

# ── Step 1: Load Bitext Dataset ──────────────────────────────
def load_bitext():
    print("📥 Loading Bitext Customer Support dataset from HuggingFace...")
    dataset = load_dataset(
        "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
        split="train"
    )
    print(f"   ✅ Loaded {len(dataset)} rows")
    print(f"   Columns: {dataset.column_names}")
    return dataset

# ── Step 2: Row-Level Chunking ───────────────────────────────
def build_documents(dataset):
    """
    Chunking Strategy: ROW-LEVEL CHUNKING
    - Each row's `instruction`  = 1 chunk (searchable content)
    - category + intent + response + source = metadata
    - Skipping `flags` column entirely as specified
    """
    print("\n📄 Building LangChain Documents (Row-Level Chunking)...")
    docs = []
    for row in dataset:
        doc = Document(
            # CHUNK: The customer's message — this is what FAISS searches
            page_content=row["instruction"],

            # METADATA: Attached to the chunk, retrieved alongside it
            metadata={
                "category": row["category"],
                "intent":   row["intent"],
                "response": row["response"],
                "source":   "documents/CS.csv"   # As specified in your sketch
            }
        )
        docs.append(doc)

    print(f"   ✅ Created {len(docs)} document chunks")

    # Show sample
    print(f"\n   --- Sample Chunk ---")
    print(f"   Content  : {docs[0].page_content}")
    print(f"   Category : {docs[0].metadata['category']}")
    print(f"   Intent   : {docs[0].metadata['intent']}")
    print(f"   Response : {docs[0].metadata['response'][:80]}...")
    print(f"   Source   : {docs[0].metadata['source']}")

    return docs

# ── Step 3: Build & Save FAISS Index ─────────────────────────
def build_faiss(docs):
    print("\n🔢 Building FAISS Vector Index...")
    print("   Loading embedding model (all-MiniLM-L6-v2)...")
    embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print(f"   Embedding {len(docs)} documents — this takes 3-5 mins...")
    faiss_store = FAISS.from_documents(docs, embedder)

    faiss_store.save_local(str(FAISS_INDEX_DIR))
    print(f"   ✅ FAISS index saved → {FAISS_INDEX_DIR}")
    return faiss_store

# ── Step 4: Build & Save BM25 Index ──────────────────────────
def build_bm25(docs):
    print("\n🔍 Building BM25 Keyword Index...")
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 5

    # Save BM25 to disk using pickle
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25_retriever, f)

    print(f"   ✅ BM25 index saved → {BM25_INDEX_PATH}")
    return bm25_retriever

# ── MAIN ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Customer Support RAG — Index Builder")
    print("=" * 60)

    # Check if index already exists
    if (FAISS_INDEX_DIR / "index.faiss").exists() and BM25_INDEX_PATH.exists():
        print("\n⚡ Index already exists! Skipping rebuild.")
        print("   Delete the faiss_index folder to force rebuild.")
        sys.exit(0)

    dataset = load_bitext()
    docs    = build_documents(dataset)
    build_faiss(docs)
    build_bm25(docs)

    print("\n" + "=" * 60)
    print("  ✅ Index Building Complete!")
    print("  Now run: python rag_agent.py")
    print("=" * 60)
