"""Tab 3 - live workflow visualisation + the compact stepper reused in Tab 1."""

from __future__ import annotations

from typing import Any, Dict, Set

import streamlit as st

from controlplane.state import Stage

# ordered pipeline steps (label, stage-key)
STAGE_STEPS = [
    ("Guardrails", Stage.GUARDRAILS),
    ("Cache", Stage.CACHE),
    ("Router", Stage.ROUTER),
    ("Retrieval", Stage.RETRIEVAL),
    ("Answer", Stage.ANSWER),
    ("Performance", Stage.PERFORMANCE),
    ("Responsibility", Stage.RESPONSIBILITY),
    ("Decision", Stage.AGGREGATE),
    ("HITL", Stage.HITL),
    ("Final", Stage.FINALIZE),
]

_BRANCH = {Stage.PERFORMANCE, Stage.RESPONSIBILITY}


def stage_strip_html(current: str, visited: Set[str]) -> str:
    """Horizontal chip strip - used live in Tab 1 as the pipeline advances."""
    if current in (Stage.DONE,):
        visited = visited | {s for _, s in STAGE_STEPS}
    cells = []
    for label, stage in STAGE_STEPS:
        cls = "cp-step"
        if stage == current or (stage in _BRANCH and current in _BRANCH):
            cls += " now"
        elif stage in visited:
            cls += " done"
        cells.append(f'<span class="{cls}"><span class="dot"></span>{label}</span>')
    return '<div style="display:flex;flex-wrap:wrap;gap:2px;margin:6px 0">' + "".join(cells) + "</div>"


# ---- Tab 3: the full graph -------------------------------------------------------------
_EDGES = [
    (Stage.GUARDRAILS, Stage.CACHE, ""),
    (Stage.CACHE, Stage.ROUTER, "miss"),
    (Stage.ROUTER, Stage.RETRIEVAL, ""),
    (Stage.RETRIEVAL, Stage.ANSWER, ""),
    (Stage.ANSWER, Stage.PERFORMANCE, "∥"),
    (Stage.ANSWER, Stage.RESPONSIBILITY, "∥"),
    (Stage.PERFORMANCE, Stage.AGGREGATE, ""),
    (Stage.RESPONSIBILITY, Stage.AGGREGATE, ""),
    (Stage.AGGREGATE, Stage.HITL, "need info"),
    (Stage.AGGREGATE, Stage.RETRIEVAL, "retry ×1"),
    (Stage.AGGREGATE, Stage.FINALIZE, "safe / harmful"),
    (Stage.HITL, Stage.RETRIEVAL, "resume ×1"),
]
_LABELS = dict(STAGE_STEPS)
_LABELS[Stage.CACHE] = "Semantic Cache"
_LABELS[Stage.ROUTER] = "RAG Router\\n(main agent)"
_LABELS[Stage.RETRIEVAL] = "Retrieval + RRF"
_LABELS[Stage.ANSWER] = "Answer\\n(streamed)"
_LABELS[Stage.AGGREGATE] = "Decision"
_LABELS[Stage.HITL] = "Human-in-the-Loop"
_LABELS[Stage.FINALIZE] = "Final Answer"


def _dot(current: str, visited: Set[str]) -> str:
    def fill(stage):
        if stage == current or (stage in _BRANCH and current in _BRANCH):
            return '"#3b5bdb", fontcolor="white", color="#7aa2ff", penwidth=2.4'
        if stage in visited:
            return '"#123a2e", fontcolor="#7ff0c2", color="#1f6b52"'
        return '"#131d33", fontcolor="#7183a6", color="#2c3e63"'

    lines = [
        'digraph G {',
        'bgcolor="transparent"; rankdir=TB; nodesep=0.35; ranksep=0.45;',
        'node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, margin="0.16,0.09"];',
        'edge [fontname="Helvetica", fontsize=9, fontcolor="#7a8bb0", color="#37456b"];',
    ]
    ids = {}
    for i, (_, stage) in enumerate(STAGE_STEPS):
        nid = f"n{i}"
        ids[stage] = nid
        lines.append(f'{nid} [label="{_LABELS.get(stage, stage)}", fillcolor={fill(stage)}];')
    for a, b, lbl in _EDGES:
        if a in ids and b in ids:
            dashed = " style=dashed" if lbl in ("retry ×1", "resume ×1") else ""
            lines.append(f'{ids[a]} -> {ids[b]} [label="{lbl}"{dashed}];')
    lines.append("}")
    return "\n".join(lines)


def render_workflow(progress: Dict[str, Any], state: Dict[str, Any]) -> None:
    current = progress.get("stage", Stage.START)
    visited: Set[str] = set(progress.get("visited", [])) | set(state.get("stages_visited", []))
    done = current == Stage.DONE or bool(state.get("final_decision"))

    label = "Idle — run a query in the Query tab" if current in (Stage.START,) else (
        f"✅ Completed — {state.get('final_decision', '').upper()}" if done else f"▶ Running: {current}")
    st.markdown(f"### {label}")
    st.markdown(stage_strip_html(current if not done else Stage.DONE, visited), unsafe_allow_html=True)

    st.graphviz_chart(_dot(current if not done else "", visited), use_container_width=True)
    st.caption("The live view advances in the **Query / Answer** tab while a query runs; "
               "this tab shows the exact path the last query took.")

    if state:
        chips = [
            ("Route", state.get("selected_kb", "—")),
            ("Performance", state.get("perf_verdict", "—")),
            ("Responsibility", state.get("resp_status", "—")),
            ("Decision", str(state.get("final_decision", "—")).upper()),
        ]
        cells = "".join(
            f"<div style='flex:1;min-width:130px;background:#0f1626;border:1px solid #1d2842;"
            f"border-radius:10px;padding:9px 12px'>"
            f"<div style='color:#7183a6;font-size:10px;text-transform:uppercase'>{k}</div>"
            f"<div style='color:#e8eefc;font-size:15px;font-weight:700;word-break:break-word'>{v}</div></div>"
            for k, v in chips
        )
        st.markdown(f"<div style='display:flex;gap:8px;flex-wrap:wrap;margin:6px 0'>{cells}</div>",
                    unsafe_allow_html=True)

        tim = state.get("node_timings", {}) or {}
        if tim:
            import pandas as pd
            import plotly.express as px

            df = pd.DataFrame(sorted(tim.items(), key=lambda x: -x[1]), columns=["node", "ms"])
            fig = px.bar(df, x="ms", y="node", orientation="h", height=300, text="ms")
            fig.update_layout(margin=dict(l=8, r=8, t=8, b=8), paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", font_color="#9fb0d0")
            st.plotly_chart(fig, use_container_width=True)

        st.caption("path: " + "  →  ".join(state.get("stages_visited", []) or ["—"]))
