import asyncio

import pytest

from controlplane.config import KB_IDS
from controlplane.retrievers import get_kb, hybrid_retrieve
from controlplane.retrievers.base import RetrievedChunk, rrf_fuse

_QUERIES = {
    "customer_support": "how do I get a refund for my order",
    "hr_policy": "how many sick leaves am I entitled to",
    "internal_knowledge": "how to map a custom domain in azure app service",
    "toxicity_kb": "is this statement a harmful stereotype about a group",
    "decision_support": "what target cost did the team agree on for the remote",
}


def test_rrf_prefers_items_in_both_lists():
    a = [RetrievedChunk(chunk_id="x", text="x"), RetrievedChunk(chunk_id="y", text="y")]
    b = [RetrievedChunk(chunk_id="y", text="y"), RetrievedChunk(chunk_id="z", text="z")]
    fused = rrf_fuse(a, b, top_k=3)
    assert fused[0].chunk_id == "y"
    assert set(fused[0].retrieval_types) == {"vector", "bm25"}


@pytest.mark.parametrize("kb_id", KB_IDS)
def test_kb_hybrid_returns_five_fused_chunks(kb_id):
    kb = get_kb(kb_id)
    out = asyncio.run(hybrid_retrieve(kb, _QUERIES[kb_id]))
    assert 1 <= len(out["vector_chunks"]) <= 5
    assert len(out["bm25_chunks"]) >= 1
    assert 1 <= len(out["rrf_chunks"]) <= 5
    assert all(c["text"] for c in out["rrf_chunks"])
