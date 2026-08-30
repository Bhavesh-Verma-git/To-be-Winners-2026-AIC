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
    _stat_row([
        ("Total latency", f"{lat:.2f} s"),
        ("Decision", str(state.get("final_decision", "—")).upper()),
        ("Route", state.get("selected_kb", "—")),
        ("Model", str(state.get("model_used", "—"))),
        ("Cost (USD)", f"${state.get('cost_usd', 0):.5f}  (Groq = $0)"),
        ("Retries / HITL", f"{state.get('retry_count', 0)} / {state.get('hitl_count', 0)}"),
    ])

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
        ed = (state.get("entity_drift", {}) or {}).get("entity_drift_results", {})
        st.caption(f"**verdict `{state.get('perf_verdict', '—')}`**  ·  votes {state.get('detector_votes', {})}")
        st.caption(f"entity drift {ed.get('drift_score', 0)} · hallucinated {ed.get('hallucinated_entities', [])[:6]}")
        if state.get("perf_reasoning"):
            st.caption(state["perf_reasoning"])

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
