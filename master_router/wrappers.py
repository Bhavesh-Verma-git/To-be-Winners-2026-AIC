"""
============================================================
  ControlPlane.ai — Agent Wrappers
  
  Each wrapper is a thin adapter that:
  1. Loads its indexes ONCE at module-import time (singleton)
  2. Exposes a single function: invoke(query) -> RAGAgentOutput
  
  The Router calls these. It doesn't care about internals.
============================================================
"""

import os
import re
import sys
import pickle
import json
import warnings
import time
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

from master_router.contract import RAGAgentOutput

# ─────────────────────────────────────────────────────────────
#  Utility: strip <think>…</think> reasoning traces
# ─────────────────────────────────────────────────────────────
def _clean(raw: str) -> str:
    if "</think>" in raw:
        clean = raw.split("</think>")[-1].strip()
    else:
        clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
        clean = re.sub(r"<think>.*",          "", clean, flags=re.DOTALL)
    
    clean = clean.strip()
    if not clean:
        return "The agent was still thinking and didn't provide a final answer (hit token limit)."
    return clean


# ═════════════════════════════════════════════════════════════
#  AGENT 1 — Customer Support
# ═════════════════════════════════════════════════════════════
_cs_chain = None   # Singleton — loaded once

def _load_customer_support():
    global _cs_chain
    if _cs_chain is not None:
        return

    from langchain_community.retrievers import BM25Retriever
    try:
        from langchain_classic.retrievers import EnsembleRetriever
    except ImportError:
        try:
            from langchain.retrievers import EnsembleRetriever
        except ImportError:
            from langchain_community.retrievers import EnsembleRetriever

    BASE   = Path(__file__).parent.parent / "rag_agents" / "customer_support"
    FIDX   = BASE / "faiss_index"
    BM25_P = FIDX / "bm25_index.pkl"

    embedder    = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2",
                                        model_kwargs={"device": "cpu"})
    faiss_store = FAISS.load_local(str(FIDX), embedder,
                                   allow_dangerous_deserialization=True)
    with open(BM25_P, "rb") as f:
        bm25_ret = pickle.load(f)

    faiss_ret   = faiss_store.as_retriever(search_kwargs={"k": 5})
    bm25_ret.k  = 5
    hybrid      = EnsembleRetriever(retrievers=[faiss_ret, bm25_ret],
                                     weights=[0.5, 0.5])

    def _fmt(docs):
        parts = []
        for d in docs:
            parts.append(
                f"Intent   : {d.metadata.get('intent','general')}\n"
                f"Category : {d.metadata.get('category','SUPPORT')}\n"
                f"Answer   : {d.metadata.get('response', d.page_content)}"
            )
        return "\n\n---\n\n".join(parts)

    llm    = ChatGroq(model_name="qwen/qwen3.6-27b", temperature=0,
                      max_tokens=2048, api_key=GROQ_API_KEY)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a professional, empathetic customer support AI assistant.
Answer the customer's question using ONLY the retrieved support knowledge below.
Be concise, clear, and helpful.

Retrieved Support Knowledge:
{context}"""),
        ("human", "{question}")
    ])
    _cs_chain = (
        {"context": hybrid | _fmt, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )


def invoke_customer_support(query: str) -> RAGAgentOutput:
    _load_customer_support()
    try:
        raw = _cs_chain.invoke(query)
        ans = _clean(raw)
        return RAGAgentOutput(
            user_query=query, rag_answer=ans,
            source="rag_agents/customer_support/data",
            source_url=None,
            agent_name="Customer Support",
            route="customer_support",
            retrieved_n=5,
            has_code=False,
            error=None
        )
    except Exception as e:
        return RAGAgentOutput(
            user_query=query, rag_answer="Error in Customer Support agent.",
            source="", source_url=None, agent_name="Customer Support",
            route="customer_support", retrieved_n=0, has_code=False,
            error=str(e)
        )


# ═════════════════════════════════════════════════════════════
#  AGENT 2 — HR Policy
# ═════════════════════════════════════════════════════════════
_hr_chain = None

def _load_hr_policy():
    global _hr_chain
    if _hr_chain is not None:
        return

    BASE        = Path(__file__).parent.parent / "rag_agents" / "hr_policy"
    FIDX        = BASE / "faiss_index"
    PARENT_P    = FIDX / "parent_store.json"

    embedder    = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2",
                                        model_kwargs={"device": "cpu"})
    faiss_store = FAISS.load_local(str(FIDX), embedder,
                                   allow_dangerous_deserialization=True)
    with open(PARENT_P, "r", encoding="utf-8") as f:
        parent_store = json.load(f)

    def _retrieve(q):
        children = faiss_store.similarity_search(q, k=4)
        seen, docs = set(), []
        for child in children:
            pid = child.metadata.get("parent_id")
            if pid and pid not in seen:
                seen.add(pid)
                p = parent_store.get(pid)
                if p:
                    docs.append(Document(
                        page_content=p["content"],
                        metadata={"section_title": p["section_title"],
                                  "source": p["source"], "parent_id": pid}
                    ))
        return docs

    def _fmt(docs):
        parts = []
        for d in docs:
            parts.append(f"[Policy Section: {d.metadata.get('section_title','HR Policy')}]\n{d.page_content}")
        return "\n\n" + "─"*50 + "\n\n".join(parts)

    llm    = ChatGroq(model_name="qwen/qwen3.6-27b", temperature=0,
                      max_tokens=2048, api_key=GROQ_API_KEY)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert HR Policy assistant for Kamaiah Engineering Services Private Limited.
Answer employee questions using ONLY the HR Policy sections provided below.
Be direct, professional, and cite the relevant policy section.
If not found say: "This specific information is not covered in the HR Policy document."

HR Policy Sections:
{context}"""),
        ("human", "{question}")
    ])
    _hr_chain = (
        {"context": lambda q: _fmt(_retrieve(q)), "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )


def invoke_hr_policy(query: str) -> RAGAgentOutput:
    _load_hr_policy()
    try:
        raw = _hr_chain.invoke(query)
        ans = _clean(raw)
        return RAGAgentOutput(
            user_query=query, rag_answer=ans,
            source="rag_agents/hr_policy/documents/HR_Policy_KESPL.pdf",
            source_url=None,
            agent_name="HR Policy",
            route="hr_policy",
            retrieved_n=4,
            has_code=False,
            error=None
        )
    except Exception as e:
        return RAGAgentOutput(
            user_query=query, rag_answer="Error in HR Policy agent.",
            source="", source_url=None, agent_name="HR Policy",
            route="hr_policy", retrieved_n=0, has_code=False,
            error=str(e)
        )


# ═════════════════════════════════════════════════════════════
#  AGENT 3 — Azure App Service Internal Knowledge
# ═════════════════════════════════════════════════════════════
_az_chain      = None
_az_bm25       = None
_az_chunk_ids  = None
_az_chunk_store= None
_az_faiss      = None

def _load_azure_docs():
    global _az_chain, _az_bm25, _az_chunk_ids, _az_chunk_store, _az_faiss
    if _az_chain is not None:
        return

    from rank_bm25 import BM25Okapi

    BASE   = Path(__file__).parent.parent / "rag_agents" / "internal_knowledge"
    FIDX   = BASE / "faiss_index"
    BM25_P = FIDX / "bm25_index.pkl"
    JSONL  = FIDX / "chunks.jsonl"

    embedder    = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5",
                                        model_kwargs={"device": "cpu"},
                                        encode_kwargs={"normalize_embeddings": True})
    _az_faiss   = FAISS.load_local(str(FIDX), embedder,
                                   allow_dangerous_deserialization=True)
    with open(BM25_P, "rb") as f:
        bm25_data     = pickle.load(f)
    _az_bm25      = bm25_data["bm25"]
    _az_chunk_ids = bm25_data["chunk_ids"]

    _az_chunk_store = {}
    with open(JSONL, "r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line.strip())
            _az_chunk_store[c["chunk_id"]] = c

    def _rrf(fids, bids, k=60, n=5):
        scores = {}
        for r, cid in enumerate(fids):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + r + 1)
        for r, cid in enumerate(bids):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + r + 1)
        return sorted(scores, key=scores.get, reverse=True)[:n]

    def _retrieve(q):
        fres  = _az_faiss.similarity_search(q, k=10)
        fids  = [d.metadata["chunk_id"] for d in fres]
        bscores = _az_bm25.get_scores(q.lower().split())
        bids  = [_az_chunk_ids[i] for i in sorted(range(len(bscores)),
                  key=lambda i: bscores[i], reverse=True)[:10]]
        docs  = []
        for cid in _rrf(fids, bids):
            chunk = _az_chunk_store.get(cid)
            if chunk:
                docs.append(Document(
                    page_content=chunk["text"],
                    metadata={"chunk_id": chunk["chunk_id"],
                               "title": chunk["title"],
                               "section": chunk["section"],
                               "source_url": chunk["source_url"],
                               "source": chunk["source"],
                               "has_code": chunk["has_code"]}
                ))
        return docs

    def _fmt(docs):
        parts = []
        for d in docs:
            m = d.metadata
            parts.append(f"[{m.get('title','')} > {m.get('section','')}]\n"
                         f"[URL: {m.get('source_url','')}]\n{d.page_content}")
        return "\n\n" + ("─"*50 + "\n\n").join(parts)

    llm    = ChatGroq(model_name="qwen/qwen3.6-27b", temperature=0,
                      max_tokens=2048, api_key=GROQ_API_KEY)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert Microsoft Azure App Service technical assistant.
Answer ONLY using the documentation sections provided below.
Always end with: "Source: <URL>"
If the question is about a different service, respond: "I can only answer questions about Azure App Service."
If not found say: "This information is not in my Azure App Service documentation."

Azure App Service Documentation:
{context}"""),
        ("human", "{question}")
    ])
    _az_chain = (
        {"context": lambda q: _fmt(_retrieve(q)), "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )


def invoke_azure_docs(query: str) -> RAGAgentOutput:
    _load_azure_docs()
    try:
        raw = _az_chain.invoke(query)
        ans = _clean(raw)
        # Extract source URL from answer if present
        url = None
        m = re.search(r"Source:\s*(https?://\S+)", ans)
        if m:
            url = m.group(1)
        has_code = "```" in ans
        return RAGAgentOutput(
            user_query=query, rag_answer=ans,
            source="rag_agents/internal_knowledge/documents",
            source_url=url,
            agent_name="Azure App Service Docs",
            route="azure_docs",
            retrieved_n=5,
            has_code=has_code,
            error=None
        )
    except Exception as e:
        return RAGAgentOutput(
            user_query=query, rag_answer="Error in Azure Docs agent.",
            source="", source_url=None, agent_name="Azure App Service Docs",
            route="azure_docs", retrieved_n=0, has_code=False,
            error=str(e)
        )


# ═════════════════════════════════════════════════════════════
#  AGENT 4 — Toxicity Analysis RAG Agent
# ═════════════════════════════════════════════════════════════
_toxic_agent = None

def _load_toxicity():
    global _toxic_agent
    if _toxic_agent is not None:
        return

    toxic_dir = Path(__file__).parent.parent / "rag_agents" / "Toxic_RAG" / "Toxic_RAG"
    if str(toxic_dir) not in sys.path:
        sys.path.insert(0, str(toxic_dir))

    from rag_agent import ToxicRAGAgent

    cache_dir = toxic_dir / ".index_cache"
    dataset_path = toxic_dir / "Dataset" / "final_tox_Rag.csv"

    _toxic_agent = ToxicRAGAgent(
        dataset_path=str(dataset_path),
        cache_dir=str(cache_dir),
        groq_model="qwen/qwen3.6-27b",
        top_k=7
    )


def invoke_toxicity(query: str) -> RAGAgentOutput:
    try:
        _load_toxicity()
        res = _toxic_agent.query(query)
        ans = _clean(res.get("answer", ""))
        retrieved_docs = res.get("retrieved_chunks", [])
        return RAGAgentOutput(
            user_query=query,
            rag_answer=ans,
            source="rag_agents/Toxic_RAG/Toxic_RAG/Dataset/final_tox_Rag.csv",
            source_url=None,
            agent_name="Toxicity Analysis",
            route="toxicity",
            retrieved_n=len(retrieved_docs),
            has_code="```" in ans,
            error=None
        )
    except Exception as e:
        return RAGAgentOutput(
            user_query=query,
            rag_answer="Error in Toxicity Analysis agent.",
            source="", source_url=None, agent_name="Toxicity Analysis",
            route="toxicity", retrieved_n=0, has_code=False,
            error=str(e)
        )


# ═════════════════════════════════════════════════════════════
#  AGENT 5 — Decision Support RAG Agent
# ═════════════════════════════════════════════════════════════
_ds_agent = None
_ds_retriever = None

def _load_decision_support():
    global _ds_agent, _ds_retriever
    if _ds_agent is not None:
        return

    ds_dir = Path(__file__).parent.parent / "rag_agents" / "Decision Support Rag" / "Decision Support Rag"
    if str(ds_dir) not in sys.path:
        sys.path.insert(0, str(ds_dir))

    from retriever import HybridRetriever
    from agent import DecisionSupportAgent

    data_dir = ds_dir / "Data"
    _ds_retriever = HybridRetriever(data_dir=str(data_dir))
    _ds_retriever.load()
    _ds_agent = DecisionSupportAgent(model_name="qwen/qwen3.6-27b")


def invoke_decision_support(query: str) -> RAGAgentOutput:
    try:
        _load_decision_support()
        res = _ds_agent.answer_query(
            query=query,
            retriever=_ds_retriever,
            top_k=5,
            model_name="qwen/qwen3.6-27b"
        )
        ans = _clean(res.get("answer", ""))
        results = res.get("results", [])
        return RAGAgentOutput(
            user_query=query,
            rag_answer=ans,
            source="rag_agents/Decision Support Rag/Decision Support Rag/Data",
            source_url=None,
            agent_name="Decision Support",
            route="decision_support",
            retrieved_n=len(results),
            has_code="```" in ans,
            error=None
        )
    except Exception as e:
        return RAGAgentOutput(
            user_query=query,
            rag_answer="Error in Decision Support agent.",
            source="", source_url=None, agent_name="Decision Support",
            route="decision_support", retrieved_n=0, has_code=False,
            error=str(e)
        )



# ── Route dispatcher ──────────────────────────────────────────
ROUTE_MAP = {
    "customer_support": invoke_customer_support,
    "hr_policy":        invoke_hr_policy,
    "azure_docs":       invoke_azure_docs,
    "toxicity":         invoke_toxicity,
    "decision_support": invoke_decision_support,
}

def dispatch(route: str, query: str) -> RAGAgentOutput:
    fn = ROUTE_MAP.get(route)
    if fn is None:
        return RAGAgentOutput(
            user_query=query,
            rag_answer="I'm sorry, I couldn't identify which knowledge base to search. Could you rephrase your question?",
            source="", source_url=None, agent_name="Unknown",
            route="unknown", retrieved_n=0, has_code=False,
            error=f"Unknown route: {route}"
        )
    return fn(query)
