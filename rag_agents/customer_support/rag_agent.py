"""
============================================================
  Customer Support RAG Agent — Step 2: The RAG Agent
  Run this AFTER build_index.py has completed.
============================================================
"""

import os
import pickle
from pathlib import Path
from dotenv import load_dotenv

# ── Load API Key from .env ───────────────────────────────────
load_dotenv(Path(__file__).parent.parent.parent / ".env")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
    raise ValueError(
        "\n❌ Groq API key not set!\n"
        "   Open the .env file at: c:/Users/verma/Desktop/accenture/.env\n"
        "   Replace 'your_groq_api_key_here' with your real Groq API key.\n"
        "   Get free key at: https://console.groq.com/"
    )

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
try:
    from langchain_classic.retrievers import EnsembleRetriever
except ImportError:
    try:
        from langchain.retrievers import EnsembleRetriever
    except ImportError:
        from langchain_community.retrievers import EnsembleRetriever

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
FAISS_INDEX_DIR = BASE_DIR / "faiss_index"
BM25_INDEX_PATH = BASE_DIR / "faiss_index" / "bm25_index.pkl"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── Step 1: Load FAISS Index from disk ───────────────────────
def load_faiss():
    print("⚡ Loading FAISS index from disk...")
    embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    faiss_store = FAISS.load_local(
        str(FAISS_INDEX_DIR),
        embedder,
        allow_dangerous_deserialization=True
    )
    print("   ✅ FAISS index loaded")
    return faiss_store

# ── Step 2: Load BM25 Index from disk ────────────────────────
def load_bm25():
    print("⚡ Loading BM25 index from disk...")
    with open(BM25_INDEX_PATH, "rb") as f:
        bm25_retriever = pickle.load(f)
    print("   ✅ BM25 index loaded")
    return bm25_retriever

# ── Step 3: Build Hybrid Retriever ────────────────────────────
def build_hybrid_retriever(faiss_store, bm25_retriever):
    faiss_retriever = faiss_store.as_retriever(search_kwargs={"k": 5})
    bm25_retriever.k = 5

    hybrid = EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=[0.5, 0.5]   # 50% semantic + 50% keyword
    )
    print("   ✅ Hybrid Retriever (FAISS + BM25) ready")
    return hybrid

# ── Step 4: Format Retrieved Docs ────────────────────────────
def format_docs(retrieved_docs):
    parts = []
    for doc in retrieved_docs:
        parts.append(
            f"Intent   : {doc.metadata.get('intent', 'general')}\n"
            f"Category : {doc.metadata.get('category', 'SUPPORT')}\n"
            f"Answer   : {doc.metadata.get('response', doc.page_content)}"
        )
    return "\n\n---\n\n".join(parts)

# ── Step 5: Build RAG Chain ───────────────────────────────────
def build_rag_chain(hybrid_retriever):
    llm = ChatGroq(
        model_name="qwen/qwen3.6-27b",
        temperature=0,
        api_key=GROQ_API_KEY
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a professional, empathetic customer support AI assistant.
Answer the customer's question using ONLY the retrieved support knowledge below.
Be concise, clear, and helpful. If the exact answer is available, use it directly.

Retrieved Support Knowledge:
{context}"""),
        ("human", "{question}")
    ])

    rag_chain = (
        {
            "context":  hybrid_retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    print("   ✅ RAG Chain built with Groq llama-3.1-70b")
    return rag_chain

# ── Step 6: Run the Agent ─────────────────────────────────────
def run_agent(rag_chain, query: str):
    print(f"\n{'='*60}")
    print(f"CUSTOMER: {query}")
    print(f"{'='*60}")
    answer = rag_chain.invoke(query)
    print(f"AGENT   : {answer}")
    return answer

# ── MAIN ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Customer Support RAG Agent — Starting")
    print("=" * 60)

    if not (FAISS_INDEX_DIR / "index.faiss").exists():
        print("\n❌ Index not found! Run build_index.py first:")
        print("   python build_index.py")
        exit(1)

    faiss_store   = load_faiss()
    bm25_retriever = load_bm25()
    hybrid        = build_hybrid_retriever(faiss_store, bm25_retriever)
    rag_chain     = build_rag_chain(hybrid)

    print("\n✅ Agent Ready! Type your question (or 'quit' to exit)\n")

    while True:
        query = input("You: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not query:
            continue
        run_agent(rag_chain, query)
