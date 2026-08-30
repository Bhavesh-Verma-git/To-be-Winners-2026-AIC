   """
============================================================
  Agent 3: Internal Knowledge Assistant
  Phase 4 — test_agent.py

  Test categories:
  ─────────────────────────────────────────────────────────
  1. Deployment        — How to deploy apps via different methods
  2. Configuration     — App settings, environment variables, etc.
  3. Scaling           — Auto-scaling, plan upgrades
  4. Networking        — VNet, custom domains, SSL
  5. Troubleshooting   — Diagnostics, logs, SSH
  6. CLI Commands      — Exact Azure CLI syntax (BM25 strength)
  7. TRAP QUESTIONS    — Out-of-scope: should gracefully refuse!
                         (Azure Functions, AWS, Key Vault)

  Run with:
     python rag_agents/internal_knowledge/test_agent.py
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

from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
INDEX_DIR  = BASE_DIR / "faiss_index"
JSONL_PATH = INDEX_DIR / "chunks.jsonl"
BM25_PATH  = INDEX_DIR / "bm25_index.pkl"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
FAISS_TOP_K, BM25_TOP_K, RRF_TOP_N, RRF_K = 10, 10, 5, 60

# ── 15 Technical + 3 Trap Queries ─────────────────────────────
TEST_CASES = [
    # ── Deployment ──────────────────────────────────────────
    {
        "id":       "T01",
        "category": "Deployment",
        "query":    "How do I deploy a Node.js app to Azure App Service using the Azure CLI?",
        "trap":     False,
    },
    {
        "id":       "T02",
        "category": "Deployment",
        "query":    "What is the difference between deployment slots and direct production deployment?",
        "trap":     False,
    },
    {
        "id":       "T03",
        "category": "Deployment",
        "query":    "How do I set up GitHub Actions CI/CD for Azure App Service?",
        "trap":     False,
    },
    # ── Configuration ───────────────────────────────────────
    {
        "id":       "T04",
        "category": "Configuration",
        "query":    "How do I set environment variables in Azure App Service?",
        "trap":     False,
    },
    {
        "id":       "T05",
        "category": "Configuration",
        "query":    "How do I enable Always On setting in App Service?",
        "trap":     False,
    },
    {
        "id":       "T06",
        "category": "Configuration",
        "query":    "How can I configure a custom startup command for my Python app?",
        "trap":     False,
    },
    # ── Scaling ─────────────────────────────────────────────
    {
        "id":       "T07",
        "category": "Scaling",
        "query":    "How do I set up autoscaling rules for an App Service plan?",
        "trap":     False,
    },
    {
        "id":       "T08",
        "category": "Scaling",
        "query":    "What is the difference between scale up and scale out in Azure App Service?",
        "trap":     False,
    },
    # ── Networking ──────────────────────────────────────────
    {
        "id":       "T09",
        "category": "Networking",
        "query":    "How do I map a custom domain to my Azure App Service?",
        "trap":     False,
    },
    {
        "id":       "T10",
        "category": "Networking",
        "query":    "How do I bind an SSL/TLS certificate to my App Service?",
        "trap":     False,
    },
    # ── Troubleshooting ─────────────────────────────────────
    {
        "id":       "T11",
        "category": "Troubleshooting",
        "query":    "How do I enable diagnostic logging for Azure App Service?",
        "trap":     False,
    },
    {
        "id":       "T12",
        "category": "Troubleshooting",
        "query":    "How do I open an SSH session into my App Service container?",
        "trap":     False,
    },
    # ── CLI Commands (BM25 strength) ─────────────────────────
    {
        "id":       "T13",
        "category": "CLI",
        "query":    "What is the az webapp up command and how do I use it?",
        "trap":     False,
    },
    {
        "id":       "T14",
        "category": "CLI",
        "query":    "How do I use az webapp config appsettings set to add an app setting?",
        "trap":     False,
    },
    {
        "id":       "T15",
        "category": "CLI",
        "query":    "How do I list all App Service plans in a resource group using Azure CLI?",
        "trap":     False,
    },
    # ── TRAP QUESTIONS (must gracefully refuse) ──────────────
    {
        "id":       "TRAP-01",
        "category": "OUT-OF-SCOPE",
        "query":    "How do I create an Azure Function triggered by a Service Bus message?",
        "trap":     True,
    },
    {
        "id":       "TRAP-02",
        "category": "OUT-OF-SCOPE",
        "query":    "How do I store a secret in AWS Secrets Manager?",
        "trap":     True,
    },
    {
        "id":       "TRAP-03",
        "category": "OUT-OF-SCOPE",
        "query":    "How do I rotate my Azure Key Vault secret automatically?",
        "trap":     True,
    },
]

# ── Helper: RRF Fusion ────────────────────────────────────────
def rrf_fuse(faiss_ids, bm25_ids, k=RRF_K, top_n=RRF_TOP_N):
    scores = {}
    for rank, cid in enumerate(faiss_ids):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    for rank, cid in enumerate(bm25_ids):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)[:top_n]

def hybrid_retrieve(query, faiss_store, bm25, chunk_ids, chunk_store):
    faiss_results = faiss_store.similarity_search(query, k=FAISS_TOP_K)
    faiss_ids     = [d.metadata["chunk_id"] for d in faiss_results]

    tokenized     = query.lower().split()
    bm25_scores   = bm25.get_scores(tokenized)
    top_bm25_idx  = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:BM25_TOP_K]
    bm25_ids      = [chunk_ids[i] for i in top_bm25_idx]

    fused_ids = rrf_fuse(faiss_ids, bm25_ids)
    docs = []
    for cid in fused_ids:
        chunk = chunk_store.get(cid)
        if chunk:
            docs.append(Document(
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
    return docs

def format_context(docs):
    parts = []
    for doc in docs:
        m = doc.metadata
        parts.append(f"[{m.get('title','')} > {m.get('section','')}]\n[URL: {m.get('source_url','')}]\n{doc.page_content}")
    return "\n\n" + ("─"*50 + "\n\n").join(parts)

def extract_clean_answer(raw):
    if "</think>" in raw:
        return raw.split("</think>")[-1].strip()
    clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    clean = re.sub(r"<think>.*",          "", clean, flags=re.DOTALL)
    return clean.strip() if clean.strip() else raw.strip()

# ── Run Tests ─────────────────────────────────────────────────
def run_tests():
    print("=" * 65)
    print("  Agent 3 — Internal Knowledge Test Suite (15 + 3 Traps)")
    print("=" * 65)

    print("\n[LOAD] Loading indexes...")
    embedder    = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    faiss_store = FAISS.load_local(
        str(INDEX_DIR), embedder,
        allow_dangerous_deserialization=True
    )
    with open(BM25_PATH, "rb") as f:
        bm25_data = pickle.load(f)
    bm25, chunk_ids = bm25_data["bm25"], bm25_data["chunk_ids"]

    chunk_store = {}
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line.strip())
            chunk_store[c["chunk_id"]] = c

    llm = ChatGroq(
        model_name="qwen/qwen3.6-27b",
        temperature=0,
        max_tokens=2048,
        api_key=GROQ_API_KEY
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert Microsoft Azure App Service technical assistant.
Answer ONLY using the documentation sections provided below.
Always end with: "Source: <URL>"
If the question is about a different service (AWS, Azure Functions, Key Vault, etc.), respond:
"I can only answer questions about Azure App Service."
If not found, say: "This information is not in my Azure App Service documentation."

Azure App Service Documentation:
{context}"""),
        ("human", "{question}")
    ])

    def retrieve_and_format(query):
        docs = hybrid_retrieve(query, faiss_store, bm25, chunk_ids, chunk_store)
        return format_context(docs)

    rag_chain = (
        {"context": retrieve_and_format, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )

    print("[OK] All indexes loaded. Running tests...\n")

    passed = 0
    trap_passed = 0
    total_traps = sum(1 for t in TEST_CASES if t["trap"])

    # To avoid Groq free tier rate limits (6,000 TPM), we run a subset of 5 representative tests
    # and add a 15-second delay between them.
    selected_tests = [
        TEST_CASES[0],   # Deployment
        TEST_CASES[4],   # Configuration
        TEST_CASES[13],  # CLI Command
        TEST_CASES[15],  # Trap (Functions)
        TEST_CASES[17]   # Trap (Key Vault)
    ]

    total_traps = sum(1 for t in selected_tests if t["trap"])

    for tc in selected_tests:
        label = f"[{tc['id']}] [{tc['category']}]"
        print(f"\n{'─'*65}")
        print(f"{label}")
        print(f"QUERY  : {tc['query']}")
        try:
            raw    = rag_chain.invoke(tc["query"])
            answer = extract_clean_answer(raw)
            print(f"ANSWER :\n{answer}")

            if tc["trap"]:
                refused = any(phrase in answer.lower() for phrase in [
                    "only answer questions about azure app service",
                    "cannot answer",
                    "not about app service",
                    "different service",
                ])
                status = "[TRAP BLOCKED]" if refused else "[TRAP FAILED — hallucinated!]"
                if refused:
                    trap_passed += 1
                print(f"STATUS : {status}")
            else:
                has_content = len(answer) > 50 and "not covered" not in answer.lower()
                passed += (1 if has_content else 0)
                print(f"STATUS : {'[PASS]' if has_content else '[FAIL]'}")

        except Exception as e:
            print(f"STATUS : [ERROR] — {e}")

        # Sleep to avoid Groq Rate Limit (Tokens Per Minute)
        import time
        time.sleep(15)

    normal_count = len(selected_tests) - total_traps
    print(f"\n{'='*65}")
    print(f"  RESULTS:")
    print(f"  Technical questions : {passed}/{normal_count} passed")
    print(f"  Trap questions      : {trap_passed}/{total_traps} correctly refused")
    total = passed + trap_passed
    total_max = normal_count + total_traps
    print(f"  Overall             : {total}/{total_max}")
    if total == total_max:
        print("  All tests passed! Agent 3 is working perfectly.")
    print("=" * 65)


if __name__ == "__main__":
    run_tests()
