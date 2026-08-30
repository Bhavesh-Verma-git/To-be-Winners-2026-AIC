"""
============================================================
  HR Policy RAG Agent — Step 2: The RAG Agent
  Uses Parent-Child retrieval:
    1. FAISS finds matching CHILD chunk (precise search)
    2. System fetches linked PARENT chunk (full context)
    3. LLM answers using the full parent section
============================================================
"""

import os
import re
import sys
import json
import warnings
from pathlib import Path
from dotenv import load_dotenv

os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

warnings.filterwarnings("ignore")
load_dotenv(Path(__file__).parent.parent.parent / ".env")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
    raise ValueError(
        "\n❌ Groq API key not set!\n"
        "   Open .env at: c:/Users/verma/Desktop/accenture/.env\n"
        "   Get free key at: https://console.groq.com/"
    )

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
FAISS_INDEX_DIR = BASE_DIR / "faiss_index"
PARENT_STORE    = BASE_DIR / "faiss_index" / "parent_store.json"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── Step 1: Load FAISS (Child Index) ─────────────────────────
def load_faiss():
    print("⚡ Loading FAISS child index...")
    embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    store = FAISS.load_local(
        str(FAISS_INDEX_DIR),
        embedder,
        allow_dangerous_deserialization=True
    )
    print("   ✅ FAISS index loaded")
    return store

# ── Step 2: Load Parent Store ─────────────────────────────────
def load_parent_store():
    print("⚡ Loading parent section store...")
    with open(PARENT_STORE, "r", encoding="utf-8") as f:
        parents = json.load(f)
    print(f"   ✅ Parent store loaded ({len(parents)} sections)")
    return parents

# ── Step 3: Parent-Child Retriever ───────────────────────────
class ParentChildRetriever:
    def __init__(self, faiss_store, parent_store, k=4):
        self.faiss_store  = faiss_store
        self.parent_store = parent_store
        self.k            = k

    def invoke(self, query: str):
        child_results = self.faiss_store.similarity_search(query, k=self.k)
        seen_parents = set()
        parent_docs  = []

        for child in child_results:
            parent_id = child.metadata.get("parent_id")
            if parent_id and parent_id not in seen_parents:
                seen_parents.add(parent_id)
                parent_data = self.parent_store.get(parent_id)
                if parent_data:
                    parent_docs.append(Document(
                        page_content=parent_data["content"],
                        metadata={
                            "section_title": parent_data["section_title"],
                            "source":        parent_data["source"],
                            "parent_id":     parent_id
                        }
                    ))

        return parent_docs

# ── Step 4: Format Context for LLM ───────────────────────────
def format_docs(retrieved_docs):
    parts = []
    for doc in retrieved_docs:
        parts.append(
            f"[Policy Section: {doc.metadata.get('section_title', 'HR Policy')}]\n"
            f"{doc.page_content}"
        )
    return "\n\n" + "─"*50 + "\n\n".join(parts)

# ── Step 5: Clean Answer from Thinking Models ─────────────────
def extract_clean_answer(raw_text: str) -> str:
    """Removes <think>...</think> reasoning traces from reasoning models."""
    if "</think>" in raw_text:
        return raw_text.split("</think>")[-1].strip()
    clean = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL)
    clean = re.sub(r"<think>.*", "", clean, flags=re.DOTALL)
    return clean.strip() if clean.strip() else raw_text.strip()

# ── Step 6: Build RAG Chain ───────────────────────────────────
def build_rag_chain(retriever):
    llm = ChatGroq(
        model_name="qwen/qwen3.6-27b",
        temperature=0,
        max_tokens=2048,
        api_key=GROQ_API_KEY
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert HR Policy assistant for Kamaiah Engineering Services Private Limited.
Answer employee questions using ONLY the HR Policy sections provided below.
Be direct, professional, and cite the relevant policy section in your answer.
If the answer is not in the provided sections, say: "This specific information is not covered in the HR Policy document."

HR Policy Sections:
{context}"""),
        ("human", "{question}")
    ])

    def retrieve_and_format(query):
        docs = retriever.invoke(query)
        return format_docs(docs)

    rag_chain = (
        {
            "context":  retrieve_and_format,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    print("   ✅ HR Policy RAG Chain ready (qwen/qwen3.6-27b)")
    return rag_chain

# ── Step 7: Run Query ─────────────────────────────────────────
def run_agent(rag_chain, query: str):
    print(f"\n{'='*62}")
    print(f"EMPLOYEE: {query}")
    print(f"{'='*62}")
    raw_answer = rag_chain.invoke(query)
    clean_ans  = extract_clean_answer(raw_answer)
    print(f"HR BOT  :\n{clean_ans}")
    return clean_ans

# ── MAIN ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 62)
    print("  HR Policy RAG Agent — Kamaiah Engineering Services")
    print("=" * 62)

    if not FAISS_INDEX_DIR.exists() or not PARENT_STORE.exists():
        print("\n❌ Index not found! Run build_index.py first:")
        print("   python build_index.py")
        exit(1)

    faiss_store   = load_faiss()
    parent_store  = load_parent_store()
    retriever     = ParentChildRetriever(faiss_store, parent_store, k=4)
    rag_chain     = build_rag_chain(retriever)

    print("\n✅ HR Policy Agent Ready!")
    print("   Ask any HR question (or type 'quit' to exit)\n")

    while True:
        query = input("You: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not query:
            continue
        run_agent(rag_chain, query)
