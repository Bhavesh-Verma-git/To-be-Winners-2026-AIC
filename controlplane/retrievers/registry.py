"""
Model + knowledge-base singletons.

Everything expensive (embedders, FAISS/Chroma indexes, BM25 pickles) is loaded
once and cached here, so per-query retrieval touches only warm objects.

`get_kb(kb_id)` returns a ready `HybridKB` adapter. `warm_all()` (used by the
warmup script and the Streamlit app) forces every adapter to load up-front.
"""

from __future__ import annotations

import threading
from functools import lru_cache
from typing import Dict, List, Optional

from controlplane.config import KB_IDS, settings

_lock = threading.Lock()


# --------------------------------------------------------------------------------------
# Embedders
# --------------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_minilm():
    """Raw SentenceTransformer('all-MiniLM-L6-v2') - cache, guardrail sim, decision-support."""
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(settings.minilm_model, device=settings.embed_device)
    except Exception:
        return None


def _lc_hf_embeddings(model_name: str, normalize: bool):
    try:
        from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
    except Exception:
        from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": settings.embed_device},
        encode_kwargs={"normalize_embeddings": normalize},
    )


@lru_cache(maxsize=2)
def get_lc_minilm(normalize: bool = False):
    """LangChain embeddings wrapper for MiniLM (matches how each FAISS index was built)."""
    return _lc_hf_embeddings(settings.minilm_model, normalize)


@lru_cache(maxsize=1)
def get_lc_bge():
    return _lc_hf_embeddings(settings.bge_model, normalize=True)


# --------------------------------------------------------------------------------------
# Knowledge bases
# --------------------------------------------------------------------------------------
_KB_CACHE: Dict[str, object] = {}


def _make_kb(kb_id: str):
    if kb_id == "customer_support":
        from controlplane.retrievers.customer_support import CustomerSupportKB

        return CustomerSupportKB()
    if kb_id == "hr_policy":
        from controlplane.retrievers.hr_policy import HRPolicyKB

        return HRPolicyKB()
    if kb_id == "internal_knowledge":
        from controlplane.retrievers.internal_knowledge import InternalKnowledgeKB

        return InternalKnowledgeKB()
    if kb_id == "toxicity_kb":
        from controlplane.retrievers.toxicity_kb import ToxicityKB

        return ToxicityKB()
    if kb_id == "decision_support":
        from controlplane.retrievers.decision_support import DecisionSupportKB

        return DecisionSupportKB()
    raise KeyError(f"Unknown knowledge base: {kb_id}")


def get_kb(kb_id: str):
    kb_id = (kb_id or "").strip()
    if kb_id in _KB_CACHE:
        return _KB_CACHE[kb_id]
    with _lock:
        if kb_id not in _KB_CACHE:
            kb = _make_kb(kb_id)
            kb.load()
            _KB_CACHE[kb_id] = kb
    return _KB_CACHE[kb_id]


def list_kbs() -> List[str]:
    return list(KB_IDS)


def warm_all(verbose: bool = False) -> Dict[str, str]:
    import time

    status: Dict[str, str] = {}
    for kb_id in KB_IDS:
        t0 = time.perf_counter()
        try:
            get_kb(kb_id)
            status[kb_id] = f"ok ({(time.perf_counter() - t0):.1f}s)"
        except Exception as exc:  # noqa: BLE001
            status[kb_id] = f"FAILED: {exc}"
        if verbose:
            print(f"  [{kb_id}] {status[kb_id]}")
    return status
