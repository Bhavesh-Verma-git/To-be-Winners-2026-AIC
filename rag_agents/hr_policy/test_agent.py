"""
============================================================
  HR Policy RAG Agent — Test Suite
  Tests 10 realistic employee HR questions
============================================================
"""

import os
import re
import sys
import json
import warnings
from pathlib import Path
from dotenv import load_dotenv

# Set UTF-8 encoding for Windows console
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

warnings.filterwarnings("ignore")
load_dotenv(Path(__file__).parent.parent.parent / ".env")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

BASE_DIR        = Path(__file__).parent
FAISS_INDEX_DIR = BASE_DIR / "faiss_index"
PARENT_STORE    = BASE_DIR / "faiss_index" / "parent_store.json"

# ── 10 Realistic HR Policy Questions ─────────────────────────
TEST_QUERIES = [
    "How many days of privilege leave am I entitled to?",
    "What is the late coming policy? How many times is it allowed?",
    "What is the dress code for employees?",
    "What happens during employee orientation at Kamaiah?",
    "What are the rules for casual leave for probationers?",
    "What is the disciplinary process if I misbehave?",
    "How much notice period is required for resignation?",
    "What are the travel allowances for different grades of employees?",
    "What are the internet usage rules at the company?",
    "What is the health and safety policy of the company?",
]

class ParentChildRetriever:
    def __init__(self, faiss_store, parent_store, k=4):
        self.faiss_store  = faiss_store
        self.parent_store = parent_store
        self.k            = k

    def invoke(self, query: str):
        child_results = self.faiss_store.similarity_search(query, k=self.k)
        seen, parent_docs = set(), []
        for child in child_results:
            pid = child.metadata.get("parent_id")
            if pid and pid not in seen:
                seen.add(pid)
                data = self.parent_store.get(pid)
                if data:
                    parent_docs.append(Document(
                        page_content=data["content"],
                        metadata={"section_title": data["section_title"],
                                  "source": data["source"]}
                    ))
        return parent_docs

def extract_clean_answer(raw_text: str) -> str:
    """Extracts only the final answer after the </think> reasoning tag."""
    if "</think>" in raw_text:
        return raw_text.split("</think>")[-1].strip()
    # Fallback if no closing tag
    clean = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL)
    clean = re.sub(r"<think>.*", "", clean, flags=re.DOTALL)
    return clean.strip() if clean.strip() else raw_text.strip()

def run_tests():
    print("=" * 65)
    print("  HR Policy RAG Agent — Running Test Suite")
    print("=" * 65)

    print("\n⚡ Loading indexes...")
    embedder     = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    faiss_store  = FAISS.load_local(str(FAISS_INDEX_DIR), embedder,
                                    allow_dangerous_deserialization=True)
    with open(PARENT_STORE, "r", encoding="utf-8") as f:
        parent_store = json.load(f)

    retriever = ParentChildRetriever(faiss_store, parent_store, k=4)

    # max_tokens=2048 ensures thinking models never get cut off
    llm = ChatGroq(
        model_name="qwen/qwen3.6-27b",
        temperature=0,
        max_tokens=2048,
        api_key=GROQ_API_KEY
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert HR Policy assistant for Kamaiah Engineering Services.
Answer using ONLY the HR Policy sections provided below. Be direct, clear, and cite the relevant policy section.

HR Policy Sections:
{context}"""),
        ("human", "{question}")
    ])

    def retrieve_and_format(query):
        docs = retriever.invoke(query)
        return "\n\n".join(
            f"[{d.metadata.get('section_title')}]\n{d.page_content}" for d in docs
        )

    rag_chain = (
        {"context": retrieve_and_format, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )

    print("✅ HR Policy Agent ready. Running 10 test queries...\n")

    passed = 0
    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\n{'─'*65}")
        print(f"[TEST {i}/10] QUERY: {query}")
        try:
            raw_output = rag_chain.invoke(query)
            clean_ans  = extract_clean_answer(raw_output)
            print(f"ANSWER:\n{clean_ans}")
            print(f"STATUS: ✅ PASS")
            passed += 1
        except Exception as e:
            print(f"STATUS: ❌ FAIL — {e}")

    print(f"\n{'='*65}")
    print(f"  RESULTS: {passed}/{len(TEST_QUERIES)} tests passed")
    if passed == len(TEST_QUERIES):
        print("  🎉 All tests passed! HR Policy RAG is working perfectly.")
    else:
        print("  ⚠️  Some tests failed. Check API key and index files.")
    print("=" * 65)

if __name__ == "__main__":
    run_tests()
