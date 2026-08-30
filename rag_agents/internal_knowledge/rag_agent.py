"""
============================================================
  Agent 3: Internal Knowledge Assistant
  Phase 3 — rag_agent.py

  What this script does:
  ─────────────────────
  Uses a hybrid retrieval strategy:

  1. FAISS  → Semantic search (finds by MEANING)
              "How to make my app scale automatically?"
              → finds "autoscale" sections even if word differs

  2. BM25   → Keyword search (finds EXACT technical terms)
              "az webapp config appsettings set"
              → finds exact CLI command string

  3. RRF    → Reciprocal Rank Fusion
              Merges FAISS + BM25 results using rank positions
              (not raw scores). This is smarter than a fixed
              50/50 weight because it adapts to query type.

  4. LLM    → qwen/qwen3.6-27b on Groq
              Generates grounded answer with source URL citation.
              Will explicitly refuse out-of-scope questions.

  Interactive loop:
     python rag_agents/internal_knowledge/rag_agent.py
============================================================
"""

import os
import re
import sys
import json
import pickle
import warnings
from pathlib import Path
from dotenv import load_dotenv

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

load_dotenv(Path(__file__).parent.parent.parent / ".env")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
    raise ValueError(
        "\n[ERROR] GROQ_API_KEY not found in .env file!\n"
        "        Get a free key at: https://console.groq.com/"
    )

from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
INDEX_DIR   = BASE_DIR / "faiss_index"
JSONL_PATH  = INDEX_DIR / "chunks.jsonl"
BM25_PATH   = INDEX_DIR / "bm25_index.pkl"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# ── Retrieval Settings ────────────────────────────────────────
FAISS_TOP_K = 10   # Fetch top 10 from semantic search
BM25_TOP_K  = 10   # Fetch top 10 from keyword search
RRF_TOP_N   = 5    # After RRF fusion, return top 5 final chunks
RRF_K       = 60   # RRF constant (standard default, proven in literature)


# ── Step 1: Load All Indexes ──────────────────────────────────
def load_indexes():
    print("[LOAD] Loading FAISS vector index...")
    embedder    = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    faiss_store = FAISS.load_local(
        str(INDEX_DIR), embedder,
        allow_dangerous_deserialization=True
    )
    print(f"       FAISS loaded from: {INDEX_DIR}")

    print("[LOAD] Loading BM25 keyword index...")
    with open(BM25_PATH, "rb") as f:
        bm25_data = pickle.load(f)
    bm25      = bm25_data["bm25"]
    chunk_ids = bm25_data["chunk_ids"]
    print(f"       BM25 loaded ({len(chunk_ids)} chunks)")

    print("[LOAD] Loading chunk JSONL store...")
    chunk_store = {}
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line.strip())
            chunk_store[chunk["chunk_id"]] = chunk
    print(f"       JSONL loaded ({len(chunk_store)} chunks)")

    return faiss_store, bm25, chunk_ids, chunk_store


# ── Step 2: RRF Fusion ────────────────────────────────────────
def rrf_fuse(
    faiss_ids: list[str],
    bm25_ids:  list[str],
    k:         int = RRF_K,
    top_n:     int = RRF_TOP_N
) -> list[str]:
    """
    Reciprocal Rank Fusion (RRF) — merges two ranked lists.

    How it works:
    - For each result in each list, compute: score = 1 / (k + rank)
    - A chunk appearing in BOTH lists gets scores from BOTH added together
    - Sort by combined score (highest first)
    - Return top_n chunk_ids

    Why k=60?
    - k is a smoothing constant. k=60 is the proven sweet spot from the
      original RRF paper — small enough to reward high ranks, large enough
      to not over-penalize low ranks.

    Example:
      FAISS: [doc_A rank=1, doc_B rank=2, doc_C rank=3]
      BM25:  [doc_B rank=1, doc_D rank=2, doc_A rank=3]

      doc_A: 1/(60+1) + 1/(60+3) = 0.01639 + 0.01563 = 0.03202
      doc_B: 1/(60+2) + 1/(60+1) = 0.01613 + 0.01639 = 0.03252  ← Winner
      doc_C: 1/(60+3) + 0        = 0.01563
      doc_D: 0 + 1/(60+2)        = 0.01613
    """
    scores = {}

    for rank, cid in enumerate(faiss_ids):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)

    for rank, cid in enumerate(bm25_ids):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)

    # Sort by combined RRF score, return top_n
    ranked = sorted(scores, key=scores.get, reverse=True)
    return ranked[:top_n]


# ── Step 3: Hybrid Retrieve ────────────────────────────────────
def hybrid_retrieve(
    query:       str,
    faiss_store,
    bm25:        BM25Okapi,
    chunk_ids:   list[str],
    chunk_store: dict
) -> list[Document]:
    """
    Runs both FAISS and BM25 searches, then fuses with RRF.
    Returns the top-N chunks as LangChain Documents.
    """

    # ── FAISS Semantic Search ────────────────────────────────
    faiss_results  = faiss_store.similarity_search(query, k=FAISS_TOP_K)
    faiss_hit_ids  = [doc.metadata["chunk_id"] for doc in faiss_results]

    # ── BM25 Keyword Search ──────────────────────────────────
    tokenized_query = query.lower().split()
    bm25_scores     = bm25.get_scores(tokenized_query)
    # Get top-K chunk indices sorted by score (highest first)
    top_bm25_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:BM25_TOP_K]
    bm25_hit_ids = [chunk_ids[i] for i in top_bm25_indices]

    # ── RRF Fusion ───────────────────────────────────────────
    fused_ids = rrf_fuse(faiss_hit_ids, bm25_hit_ids)

    # ── Build LangChain Documents from fused results ─────────
    result_docs = []
    for cid in fused_ids:
        chunk = chunk_store.get(cid)
        if chunk:
            result_docs.append(Document(
                page_content=chunk["text"],
                metadata={
                    "chunk_id":   chunk["chunk_id"],
                    "title":      chunk["title"],
                    "section":    chunk["section"],
                    "source_url": chunk["source_url"],
                    "source":     chunk["source"],
                    "has_code":   chunk["has_code"],
                }
            ))

    return result_docs


# ── Step 4: Format Context for LLM ────────────────────────────
def format_context(docs: list[Document]) -> str:
    parts = []
    for doc in docs:
        meta    = doc.metadata
        section = meta.get("section", "Azure App Service")
        title   = meta.get("title", "")
        url     = meta.get("source_url", "")
        parts.append(
            f"[Source: {title} > {section}]\n"
            f"[URL: {url}]\n"
            f"{doc.page_content}"
        )
    return "\n\n" + ("─" * 50 + "\n\n").join(parts)


# ── Step 5: Clean LLM Answer ──────────────────────────────────
def extract_clean_answer(raw: str) -> str:
    """Removes <think>...</think> reasoning traces from Qwen model."""
    if "</think>" in raw:
        return raw.split("</think>")[-1].strip()
    clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    clean = re.sub(r"<think>.*",          "", clean, flags=re.DOTALL)
    return clean.strip() if clean.strip() else raw.strip()


# ── Step 6: Build RAG Chain ────────────────────────────────────
def build_rag_chain(faiss_store, bm25, chunk_ids, chunk_store):
    llm = ChatGroq(
        model_name="qwen/qwen3.6-27b",
        temperature=0,
        max_tokens=2048,
        api_key=GROQ_API_KEY
    )

    # System prompt with strict guardrails:
    # 1. Must answer from provided context only
    # 2. Must always include the source URL
    # 3. Must include code snippets when available
    # 4. Must explicitly refuse out-of-scope questions
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert Microsoft Azure App Service technical assistant.
Your knowledge comes EXCLUSIVELY from the Azure App Service documentation sections provided below.

Rules you MUST follow:
1. Answer ONLY using the documentation sections below. Do not use outside knowledge.
2. Always end your answer with: "Source: <URL from the documentation>"
3. If a code snippet (Azure CLI, PowerShell, JSON) is available in the context, include it in your answer.
4. If the question is about a different Azure service (e.g., Azure Functions, Azure Key Vault, AWS) that is NOT App Service, respond with:
   "I can only answer questions about Azure App Service. Your question appears to be about a different service."
5. If the answer is genuinely not found in the provided sections, say:
   "This specific information is not covered in the Azure App Service documentation I have access to."

Azure App Service Documentation:
{context}"""),
        ("human", "{question}")
    ])

    def retrieve_and_format(query):
        docs = hybrid_retrieve(query, faiss_store, bm25, chunk_ids, chunk_store)
        return format_context(docs)

    rag_chain = (
        {
            "context":  retrieve_and_format,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# ── Step 7: Run One Query ──────────────────────────────────────
def run_query(rag_chain, query: str) -> str:
    print(f"\n{'='*62}")
    print(f"YOU  : {query}")
    print(f"{'='*62}")
    raw    = rag_chain.invoke(query)
    answer = extract_clean_answer(raw)
    print(f"BOT  :\n{answer}")
    return answer


# ── MAIN ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 62)
    print("  Agent 3 — Azure App Service Internal Knowledge Assistant")
    print("  Hybrid Retrieval: FAISS + BM25 + RRF")
    print("=" * 62)

    if not INDEX_DIR.exists():
        print("\n[ERROR] Index not found! Run build_index.py first:")
        print("        python rag_agents/internal_knowledge/build_index.py")
        sys.exit(1)

    faiss_store, bm25, chunk_ids, chunk_store = load_indexes()
    rag_chain = build_rag_chain(faiss_store, bm25, chunk_ids, chunk_store)

    print("\n[READY] Azure App Service Knowledge Assistant is ready!")
    print("        Ask technical questions or type 'quit' to exit.\n")

    while True:
        query = input("You: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not query:
            continue
        run_query(rag_chain, query)
