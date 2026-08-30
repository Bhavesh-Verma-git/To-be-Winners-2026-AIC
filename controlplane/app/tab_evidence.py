"""Tab 4 - Retrieval & Evidence. Every chunk the pipeline pulled + the laws implicated."""

from __future__ import annotations

import html
from typing import Any, Dict, List

import streamlit as st


def _chunk_card(i: int, c: Dict[str, Any]) -> None:
    title = html.escape(str(c.get("title") or c.get("source") or f"chunk {i}"))
    score = c.get("score", 0)
    rtypes = ", ".join(c.get("retrieval_types", []) or [])
    ranks = c.get("ranks", {}) or {}
    with st.expander(f"[{i}]  {title}   ·   RRF {score}"):
        st.markdown(
            f"<span class='cp-chip'>{rtypes or 'n/a'}</span> "
            + " ".join(f"<span class='cp-chip'>{k}:#{v}</span>" for k, v in ranks.items()),
            unsafe_allow_html=True,
        )
        md = c.get("metadata", {}) or {}
        if md.get("law_or_article"):
            st.markdown(f"**Law / Article:** `{md['law_or_article']}`")
        if md.get("source_url"):
            st.markdown(f"[source]({md['source_url']})")
        st.code((c.get("text", "") or "")[:1800], language=None)


def _section(label: str, chunks: List[Dict[str, Any]]) -> None:
    st.markdown(f"##### {label}  ·  {len(chunks or [])}")
    if not chunks:
        st.caption("— none —")
        return
    for i, c in enumerate(chunks, 1):
        _chunk_card(i, c)


def render_evidence(state: Dict[str, Any]) -> None:
    if not state:
        st.info("Run a query to see every retrieved chunk and the evidence used.")
        return

    kb = state.get("selected_kb", "—")
    dec = str(state.get("final_decision", "—")).upper()
    st.markdown(f"### Evidence for the last query  —  route `{kb}`  ·  decision **{dec}**")

    # ---- laws / rules implicated ----
    rules = state.get("violated_rules", []) or []
    resp_status = state.get("resp_status")
    if rules or resp_status in ("unsafe", "uncertain"):
        box = st.error if resp_status == "unsafe" else st.warning
        box("**Laws / rules implicated:** " + (", ".join(rules) if rules else "under review"))
    if state.get("resp_report"):
        with st.expander("📜 Full compliance analysis", expanded=(resp_status == "unsafe")):
            st.markdown(state["resp_report"])

    tox = state.get("toxicity", {}) or {}
    if tox:
        cells = "  ".join(
            f"<span class='cp-chip'>{k}: {v.get('prob')} ({v.get('label')})</span>"
            for k, v in tox.items() if isinstance(v, dict)
        )
        st.markdown(f"**Toxicity ensemble** (max {state.get('toxicity_max', 0)}): {cells}",
                    unsafe_allow_html=True)

    st.divider()

    # ---- RAG retrieval pipeline ----
    st.markdown("## 🔎 RAG retrieval pipeline")
    st.caption("Vector (FAISS / Chroma) and BM25 each return top-5, then Reciprocal Rank Fusion picks the final 5.")
    a, b = st.columns(2)
    with a:
        _section("Vector search", state.get("vector_chunks", []))
    with b:
        _section("BM25 search", state.get("bm25_chunks", []))
    _section("➡️ RRF — final context sent to the answer generator", state.get("rrf_chunks", []))

    st.divider()

    # ---- Responsibility retrieval pipeline ----
    st.markdown("## 🛡️ Responsibility retrieval pipeline")
    meta = state.get("resp_retrieval_meta", {}) or {}
    st.caption(
        f"Vector · BM25 · Knowledge-Graph (Neo4j: {meta.get('neo4j', False)}, "
        f"vector backend: {meta.get('vector_backend', 'n/a')}) → RRF. "
        "Runs on the generated answer to find EU AI Act / NIST / hate-speech clauses it may touch."
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        _section("Vector", state.get("resp_vector_chunks", []))
    with c2:
        _section("BM25", state.get("resp_bm25_chunks", []))
    with c3:
        _section("Knowledge Graph", state.get("resp_graph_chunks", []))
    _section("➡️ RRF — evidence for the responsibility verdict", state.get("resp_rrf_chunks", []))
