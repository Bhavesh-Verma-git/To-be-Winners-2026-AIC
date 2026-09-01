"""Tab 2 - hackathon dashboard. Node/model latency, cost (Gemini only), chunks, scores."""

from __future__ import annotations

import html
from typing import Any, Dict

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from controlplane.observability import fetch_run_metrics

_LAYOUT = dict(margin=dict(l=140, r=20, t=36, b=30), paper_bgcolor="rgba(0,0,0,0)",
               plot_bgcolor="rgba(0,0,0,0)", font_color="#9fb0d0", height=300,
               coloraxis_showscale=False, showlegend=False)


def _style(fig, **kw):
    fig.update_layout(**{**_LAYOUT, **kw})
    return fig


_ED_WARN, _ED_FAIL = 0.20, 0.40   # DRIFT_SCORE_WARN/FAIL thresholds (entity_drift_agent.py)


def _render_entity_drift(state: Dict[str, Any]) -> None:
    """Full Entity-Drift panel: every computed value + visualisations. All numbers
    come straight from the running EntityDriftAgent - nothing fabricated."""
    ed_all = state.get("entity_drift", {}) or {}
    ed = ed_all.get("entity_drift_results", {}) or {}
    if not ed:
        st.caption("Entity drift: no data (cache hit / guardrail block / no answer).")
        return

    ctx_ents = list(ed.get("context_entities", []) or [])
    resp_ents = list(ed.get("response_entities", []) or [])
    halluc = list(ed.get("hallucinated_entities", []) or [])            # added / not in source
    halluc_raw = list(ed.get("hallucinated_entities_raw", halluc) or [])
    matched = sorted(set(e.lower() for e in resp_ents) & set(e.lower() for e in ctx_ents))
    removed = sorted(set(e.lower() for e in ctx_ents) - set(e.lower() for e in resp_ents))
    drift = float(ed.get("drift_score", 0.0) or 0.0)
    overlap = float(ed.get("entity_overlap_ratio", 0.0) or 0.0)
    rel_pairs = ed.get("relation_drift_pairs", []) or []
    rel_score = float(ed.get("relation_drift_score", 0.0) or 0.0)
    verdict = ed_all.get("entity_drift_verdict", "—")
    reasoning = ed_all.get("entity_drift_reasoning", "")
    lat = ed_all.get("entity_drift_latency_ms", 0.0)

    st.markdown("###### 🔬 Entity drift  ·  spaCy NER (no LLM)")
    _stat_row([
        ("Drift score", f"{drift:.2f}"),
        ("Overlap ratio", f"{overlap:.2f}"),
        ("Verdict", str(verdict).upper()),
        ("Warn / Fail @", f"{_ED_WARN:.2f} / {_ED_FAIL:.2f}"),
        ("Matched / Added / Removed", f"{len(matched)} / {len(halluc)} / {len(removed)}"),
        ("Relation drift", f"{len(rel_pairs)} pair(s)  ({rel_score:.2f})"),
        ("Latency", f"{lat:.0f} ms"),
    ])

    c1, c2 = st.columns(2)
    with c1:
        # drift score vs thresholds
        g = go.Figure(go.Indicator(
            mode="gauge+number", value=drift,
            title={"text": f"drift score  (verdict: {verdict})"},
            gauge={"axis": {"range": [0, 1]}, "bar": {"color": "#7aa2ff"},
                   "steps": [{"range": [0, _ED_WARN], "color": "#0f2620"},
                             {"range": [_ED_WARN, _ED_FAIL], "color": "#2a2410"},
                             {"range": [_ED_FAIL, 1], "color": "#331515"}],
                   "threshold": {"line": {"color": "#ff8f8f", "width": 3}, "value": _ED_FAIL}}))
        st.plotly_chart(_style(g, height=240), use_container_width=True)
    with c2:
        cdf = pd.DataFrame({
            "category": ["matched (grounded)", "added (not in source)", "removed (unused source)"],
            "count": [len(matched), len(halluc), len(removed)],
        })
        fig = px.bar(cdf, x="count", y="category", orientation="h", text="count",
                     color="category",
                     color_discrete_map={"matched (grounded)": "#57e0c7",
                                         "added (not in source)": "#ff8f8f",
                                         "removed (unused source)": "#8fa3c8"})
        fig.update_traces(textposition="outside", cliponaxis=False)
        st.plotly_chart(_style(fig, height=240, showlegend=False), use_container_width=True)

    # entity-level comparison table
    rows = []
    for e in sorted(set(e.lower() for e in ctx_ents) | set(e.lower() for e in resp_ents)):
        in_ctx = e in set(x.lower() for x in ctx_ents)
        in_resp = e in set(x.lower() for x in resp_ents)
        status = ("✅ matched" if in_ctx and in_resp
                  else "🟥 added (drift)" if in_resp else "⬜ in source only")
        rows.append({"entity": e, "in source": in_ctx, "in answer": in_resp, "status": status})
    if rows:
        with st.expander(f"entity-level comparison  ({len(rows)} entities)", expanded=bool(halluc)):
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=min(320, 40 + 32 * len(rows)))
    if rel_pairs:
        st.warning("Relation drift: " + "; ".join(
            f"{p.get('entities')}: source says '{p.get('context_relation')}' but answer says '{p.get('answer_relation')}'"
            for p in rel_pairs))
    if halluc_raw and halluc_raw != halluc:
        st.caption(f"filtered as trivial (list markers / units / small cardinals): "
                   f"{sorted(set(e.lower() for e in halluc_raw) - set(e.lower() for e in halluc))}")
    if reasoning:
        st.caption(reasoning)


def _stat_row(items) -> None:
    """Full-text stat cards (st.metric truncates long values with '...')."""
    cells = "".join(
        f"<div style='flex:1;min-width:150px;background:#0f1626;border:1px solid #1d2842;"
        f"border-radius:12px;padding:10px 14px'>"
        f"<div style='color:#7183a6;font-size:11px;text-transform:uppercase;letter-spacing:.4px'>{html.escape(k)}</div>"
        f"<div style='color:#e8eefc;font-size:18px;font-weight:700;word-break:break-word;margin-top:3px'>{html.escape(str(v))}</div>"
        f"</div>"
        for k, v in items
    )
    st.markdown(f"<div style='display:flex;gap:10px;flex-wrap:wrap;margin-bottom:12px'>{cells}</div>",
                unsafe_allow_html=True)


def render_dashboard(state: Dict[str, Any], thread_id: str) -> None:
    if not state:
        st.info("Run a query in the **Query / Answer** tab to populate the dashboard.")
        return

    lat = (state.get("total_latency_ms") or 0) / 1000
    verdict = state.get("final_verdict") or str(state.get("final_decision", "—")).upper()
    _v = str(verdict).lower()
    _vcol = ("#ff8f8f" if "block" in _v else "#b9a4ff" if "edit" in _v
             else "#ffd88a" if ("human" in _v or "hitl" in _v) else "#57e0c7")
    st.markdown(
        f"<div style='font-size:20px;font-weight:800;color:{_vcol};margin-bottom:8px'>"
        f"VERDICT: {html.escape(str(verdict).upper())}</div>", unsafe_allow_html=True)
    _stat_row([
        ("Total latency", f"{lat:.2f} s"),
        ("Decision", str(state.get("final_decision", "—")).upper()),
        ("Route", f"{state.get('selected_kb', '—')}  ({state.get('router_reason', '—')})"),
        ("Model", str(state.get("model_used", "—"))),
        ("Cost (USD)", f"${state.get('cost_usd', 0):.5f}  (Groq = $0)"),
        ("Retries / HITL", f"{state.get('retry_count', 0)} / {state.get('hitl_count', 0)}"),
    ])
    sem = state.get("router_semantic_scores") or {}
    if sem:
        st.caption("router semantic scores (query ↔ each KB): "
                   + "  ".join(f"`{k}` {v}" for k, v in sorted(sem.items(), key=lambda x: -x[1])))
    if state.get("final_decision") == "harmful" or state.get("resp_status") == "unsafe":
        st.error("⚠️ **Harmful / toxic** — answer withheld. Laws implicated: "
                 + ", ".join((state.get("violated_rules") or ["see compliance report"])[:5]))
    if state.get("original_answer"):
        with st.expander("↻ Self-reflection (EDIT): original draft → revised answer", expanded=True):
            st.markdown(f"**Reason for EDIT:** {state.get('edit_reason', '—')}")
            st.markdown(f"**Rewritten retrieval query:** `{state.get('perf_suggestion', '—')}`")
            st.markdown(f"**Original draft:**\n\n> {state.get('original_answer', '')[:800]}")
            st.markdown(f"**Revised answer:**\n\n> {(state.get('answer') or '')[:800]}")

    # ---- latency ----
    st.markdown("###### Node latency (ms)")
    tim = state.get("node_timings", {}) or {}
    if tim:
        df = pd.DataFrame(sorted(tim.items(), key=lambda x: x[1]), columns=["node", "ms"])
        fig = px.bar(df, x="ms", y="node", orientation="h", text="ms")
        fig.update_traces(marker_color="#5570ff", textposition="outside", cliponaxis=False)
        st.plotly_chart(_style(fig, height=max(240, 26 * len(df) + 60)), use_container_width=True)

    st.markdown("###### Model calls  ·  latency + tokens  (Groq cost forced to $0)")
    calls = state.get("llm_calls", []) or []
    if calls:
        cdf = pd.DataFrame(calls)
        for col, d in [("node", "—"), ("category", "—"), ("model", "—"), ("provider", "—"),
                       ("tier", 0), ("prompt_tokens", 0), ("completion_tokens", 0),
                       ("latency_ms", 0), ("cost_usd", 0.0)]:
            if col not in cdf:
                cdf[col] = d
        cdf = cdf.fillna({"cost_usd": 0.0, "latency_ms": 0, "provider": "—", "model": "—"})
        cdf["cost_usd"] = cdf.apply(
            lambda r: 0.0 if str(r.get("provider", "")).lower() in ("groq", "mock")
            else (r["cost_usd"] or 0.0), axis=1)
        st.dataframe(cdf[["node", "category", "model", "tier", "prompt_tokens",
                          "completion_tokens", "latency_ms", "cost_usd"]],
                     use_container_width=True, hide_index=True, height=200)
        gem = cdf[cdf["provider"].str.lower().isin(["gemini", "vertex_ai", "other"])]["cost_usd"].sum()
        grq = int((cdf["provider"].str.lower() == "groq").sum())
        st.caption(f"**Cost:** Gemini ${gem:.5f}  ·  Groq {grq} call(s) → $0.00000")
    else:
        st.caption("no LLM calls this query (cache hit or guardrail block)")

    ls = fetch_run_metrics(state.get("langsmith_run_id"))
    if ls:
        with st.expander("🔎 LangSmith run rollup"):
            if ls.get("run_url"):
                st.markdown(f"[open in LangSmith]({ls['run_url']})")
            st.dataframe(pd.DataFrame(ls.get("models", [])), use_container_width=True, hide_index=True)

    st.divider()

    # ---- performance / responsibility ----
    p, r = st.columns(2)
    with p:
        st.markdown("### 🎯 Performance")
        rag = state.get("ragas_scores", {}) or {}
        if rag:
            k = list(rag.keys())
            fig = go.Figure(go.Scatterpolar(r=[rag[x] for x in k] + [rag[k[0]]],
                                            theta=k + [k[0]], fill="toself", line_color="#7aa2ff"))
            fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(range=[0, 1])),
                              showlegend=False, title="RAGAS")
            st.plotly_chart(_style(fig), use_container_width=True)
        prob = state.get("xgboost_prob")
        if prob is not None:
            g = go.Figure(go.Indicator(
                mode="gauge+number", value=prob * 100,
                number={"suffix": "%"},
                title={"text": f"XGBoost hallucination ({state.get('xgboost_risk', '—')})"},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#7aa2ff"},
                       "steps": [{"range": [0, 20], "color": "#0f2620"},
                                 {"range": [20, 60], "color": "#2a2410"},
                                 {"range": [60, 100], "color": "#331515"}]}))
            st.plotly_chart(_style(g), use_container_width=True)
        st.caption(f"**verdict `{state.get('perf_verdict', '—')}`**  ·  votes {state.get('detector_votes', {})}")
        if state.get("perf_reasoning"):
            st.caption(state["perf_reasoning"])
        st.divider()
        _render_entity_drift(state)

    with r:
        st.markdown("### 🛡️ Responsibility")
        tox = state.get("toxicity", {}) or {}
        if tox:
            tdf = pd.DataFrame([{"model": k, "prob": v.get("prob", 0), "label": v.get("label", "—")}
                                for k, v in tox.items()])
            st.plotly_chart(_style(px.bar(tdf, x="model", y="prob", color="label", range_y=[0, 1],
                                          title=f"Toxicity ensemble (max {state.get('toxicity_max', 0)})")),
                            use_container_width=True)
        st.caption(f"**status `{state.get('resp_status', '—')}`**")
        st.caption(state.get("resp_reasoning", ""))
        if state.get("violated_rules"):
            st.error("Rules implicated: " + ", ".join(state["violated_rules"][:6]))
        if state.get("resp_report"):
            with st.expander("compliance report"):
                st.markdown(state["resp_report"])

    st.divider()

    # ---- retrieved chunks ----
    st.markdown("### 📚 Retrieved context")
    ct = st.tabs(["RAG · vector", "RAG · BM25", "RAG · RRF (final)", "Responsibility · RRF"])
    for tab, key in zip(ct, ["vector_chunks", "bm25_chunks", "rrf_chunks", "resp_rrf_chunks"]):
        with tab:
            rows = state.get(key, []) or []
            if not rows:
                st.caption("none")
            for i, ch in enumerate(rows, 1):
                with st.expander(f"[{i}] {ch.get('title') or ch.get('source', '')}  ·  score {ch.get('score', 0)}"):
                    st.write((ch.get("text", "") or "")[:1400])
                    st.caption(f"retrieval {ch.get('retrieval_types')} · ranks {ch.get('ranks')}")

    f = st.columns(4)
    f[0].metric("Hallucination retries", state.get("retry_count", 0), help="max 1")
    f[1].metric("HITL rounds", state.get("hitl_count", 0), help="max 1")
    f[2].metric("Model tier", state.get("model_tier", "—"))
    f[3].metric("Guardrail flags", len(state.get("guardrail_flags", [])))
    if state.get("perf_suggestion"):
        st.caption(f"↻ retry query → {state['perf_suggestion']}")
