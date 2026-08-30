"""
============================================================
  Customer Support RAG Agent — Test Suite
  Run this to verify everything is working correctly.
============================================================
"""

import os
import pickle
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

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

BASE_DIR        = Path(__file__).parent
FAISS_INDEX_DIR = BASE_DIR / "faiss_index"
BM25_INDEX_PATH = BASE_DIR / "faiss_index" / "bm25_index.pkl"

# ── 10 Realistic Customer Support Test Queries ────────────────
TEST_QUERIES = [
    "How do I cancel my order?",
    "I want a refund for my purchase",
    "My account is locked, how do I log in?",
    "Where is my package? It is taking too long.",
    "How do I change my delivery address?",
    "I was charged twice for the same order",
    "How do I update my payment method?",
    "I want to delete my account",
    "The product I received is damaged",
    "How do I track my order status?",
]

def run_tests():
    print("=" * 65)
    print("  Customer Support RAG Agent — Running Test Suite")
    print("=" * 65)

    if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
        print("\n❌ Error: GROQ_API_KEY not found in .env!")
        print("   Please add your Groq API key into c:/Users/verma/Desktop/accenture/.env")
        sys.exit(1)

    print("\n⚡ Loading indexes...")
    embedder       = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    faiss_store    = FAISS.load_local(str(FAISS_INDEX_DIR), embedder,
                                      allow_dangerous_deserialization=True)
    with open(BM25_INDEX_PATH, "rb") as f:
        bm25 = pickle.load(f)
    bm25.k = 5

    hybrid = EnsembleRetriever(
        retrievers=[faiss_store.as_retriever(search_kwargs={"k": 5}), bm25],
        weights=[0.5, 0.5]
    )

    llm = ChatGroq(model_name="qwen/qwen3.6-27b", temperature=0,
                   api_key=GROQ_API_KEY)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a professional customer support AI assistant.
Answer the customer using ONLY the retrieved support knowledge below.
Retrieved Support Knowledge:\n{context}"""),
        ("human", "{question}")
    ])

    def format_docs(docs):
        return "\n\n---\n\n".join(
            f"Intent: {d.metadata.get('intent', 'general')}\nAnswer: {d.metadata.get('response', d.page_content)}"
            for d in docs
        )

    rag_chain = (
        {"context": hybrid | format_docs, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )

    print("✅ Agent ready. Running 10 test queries...\n")

    passed = 0
    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\n[TEST {i}/10]")
        print(f"  QUERY : {query}")
        try:
            answer = rag_chain.invoke(query)
            print(f"  ANSWER: {answer[:200]}{'...' if len(answer) > 200 else ''}")
            print(f"  STATUS: ✅ PASS")
            passed += 1
        except Exception as e:
            print(f"  STATUS: ❌ FAIL — {e}")

    print(f"\n{'='*65}")
    print(f"  RESULTS: {passed}/{len(TEST_QUERIES)} tests passed")
    if passed == len(TEST_QUERIES):
        print("  🎉 All tests passed! RAG Agent is working perfectly.")
    else:
        print("  ⚠️  Some tests failed. Check API key and index files.")
    print("=" * 65)

if __name__ == "__main__":
    run_tests()
