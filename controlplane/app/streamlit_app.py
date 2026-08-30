"""
ControlPlane.ai - Streamlit UI (3 tabs).

    Tab 1  Query / Answer     - ask, watch the pipeline advance + tokens stream, resolve HITL
    Tab 2  Dashboard          - node & model latency, cost (Gemini only), chunks, scores
    Tab 3  Live Workflow      - the pipeline graph with the path actually taken

Run:  streamlit run controlplane/app/streamlit_app.py
"""

from __future__ import annotations

import html
import time

import streamlit as st

from controlplane.app.runner import new_thread_id, stream_run
from controlplane.app.tab_dashboard import render_dashboard
from controlplane.app.tab_evidence import render_evidence
from controlplane.app.tab_workflow import STAGE_STEPS, render_workflow, stage_strip_html
from controlplane.cache import get_cache
from controlplane.config import settings
from controlplane.prompts import DEMO_PROMPTS
from controlplane.state import Stage

st.set_page_config(page_title="ControlPlane.ai", page_icon="🛡️", layout="wide",
                   initial_sidebar_state="expanded")

# ------------------------------------------------------------------ styling ----
st.markdown(
    """
<style>
  .stApp { background: radial-gradient(1200px 600px at 20% -10%, #14203b 0%, #0b0f1a 55%); }
  section[data-testid="stSidebar"] { background:#0d1424; border-right:1px solid #1e2b45; }
  .cp-hero {
     background: linear-gradient(110deg,#1b2a4a 0%,#131a2e 55%,#0f1626 100%);
     border:1px solid #24365c; border-radius:16px; padding:18px 22px; margin-bottom:14px;
     box-shadow:0 10px 30px -12px rgba(0,0,0,.6);
  }
  .cp-hero h1 { margin:0; font-size:22px; font-weight:800; color:#eaf0ff;
     background:linear-gradient(90deg,#7aa2ff,#a970ff,#57e0c7); -webkit-background-clip:text;
     -webkit-text-fill-color:transparent; }
  .cp-hero p { margin:4px 0 0; color:#8ea0c4; font-size:13px; }
  .cp-chip { display:inline-block; padding:3px 10px; border-radius:999px; font-size:11px;
     font-weight:600; margin:3px 5px 0 0; border:1px solid #2c3e63; color:#b9c7e6;
     background:#131d33; }
  .cp-chip.ok  { color:#7ff0c2; border-color:#1f6b52; background:#0f2a22; }
  .cp-chip.off { color:#ff9a9a; border-color:#6b2a2a; background:#2a1414; }
  .cp-badge { display:inline-block; padding:4px 11px; border-radius:8px; font-size:12px;
     font-weight:700; margin:4px 6px 0 0; letter-spacing:.3px; }
  .cp-badge.safe   { background:#0f2a22; color:#57e0c7; border:1px solid #1f6b52; }
  .cp-badge.block  { background:#2a1414; color:#ff8f8f; border:1px solid #6b2a2a; }
  .cp-badge.harm   { background:#33121f; color:#ff85b9; border:1px solid #7a2450; }
  .cp-badge.cache  { background:#12233a; color:#8fc7ff; border:1px solid #285285; }
  .cp-badge.info   { background:#161f36; color:#a9bbdf; border:1px solid #2c3e63; }
  .cp-step { display:inline-flex; align-items:center; gap:6px; font-size:11px;
     padding:6px 10px; border-radius:9px; margin:3px; border:1px solid #24365c;
     color:#7183a6; background:#101a30; white-space:nowrap; }
  .cp-step.done { color:#63d6a8; border-color:#1f6b52; background:#0f2620; }
  .cp-step.now  { color:#fff; border-color:#5570ff; background:#22336b;
     box-shadow:0 0 0 2px rgba(85,112,255,.35); font-weight:700; }
  .cp-step .dot { width:7px;height:7px;border-radius:50%;background:currentColor; }
  .stChatMessage { background:#0f1626 !important; border:1px solid #1d2842 !important;
     border-radius:14px !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ state ----
ss = st.session_state
ss.setdefault("thread_id", new_thread_id())
ss.setdefault("history", [])                 # [{role, content, badges, meta}]
ss.setdefault("final_state", {})
ss.setdefault("progress", {"stage": Stage.START, "visited": []})
ss.setdefault("pending_interrupt", None)
ss.setdefault("queued_prompt", None)


def _sysline() -> str:
    chips = []
    chips.append(f'<span class="cp-chip {"ok" if settings.has_groq() else "off"}">Groq × {len(settings.groq_keys)}</span>')
    chips.append(f'<span class="cp-chip {"ok" if settings.has_gemini() else "off"}">Gemini × {len(settings.gemini_keys)}</span>')
    chips.append(f'<span class="cp-chip {"ok" if settings.langsmith_enabled else "off"}">LangSmith</span>')
    chips.append(f'<span class="cp-chip {"ok" if settings.neo4j_uri else "off"}">Neo4j</span>')
    chips.append(f'<span class="cp-chip">{"MOCK LLM" if not settings.has_any_llm() else "LIVE"}</span>')
    chips.append(f'<span class="cp-chip">cache: {len(get_cache())}</span>')
    return " ".join(chips)


st.markdown(
    f"""<div class="cp-hero">
      <h1>🛡️ ControlPlane.ai — real-time AI governance</h1>
      <p>Guardrails · Semantic Cache · Agentic RAG · Parallel Performance ∥ Responsibility · one-shot retry / HITL · &lt;10s</p>
      <div style="margin-top:8px">{_sysline()}</div>
    </div>""",
    unsafe_allow_html=True,
)

tab_q, tab_d, tab_w, tab_e = st.tabs(
    ["💬  Query / Answer", "📊  Dashboard", "🧭  Live Workflow", "🔎  Retrieval & Evidence"]
)


# ------------------------------------------------------------------ run one query ----
def _run(prompt: str | None, resume: str | None = None) -> None:
    ss.progress = {"stage": Stage.START, "visited": []}
    ss.pending_interrupt = None

    step_ph = st.empty()
    ans_ph = st.empty()
    tokens: list[str] = []
    t0 = time.time()

    def _paint_steps():
        step_ph.markdown(stage_strip_html(ss.progress["stage"], set(ss.progress["visited"])),
                         unsafe_allow_html=True)

    _paint_steps()
    _last_paint = 0.0
    for kind, payload in stream_run(prompt, ss.thread_id, resume=resume, progress=ss.progress):
        if kind == "node":
            _paint_steps()
        elif kind == "token":
            tokens.append(payload)
            now = time.time()
            if now - _last_paint > 0.06:          # throttle re-renders
                ans_ph.markdown("".join(tokens) + " ▌")
                _last_paint = now
        elif kind == "interrupt":
            ss.pending_interrupt = payload
            step_ph.markdown(stage_strip_html(Stage.HITL, set(ss.progress["visited"])),
                             unsafe_allow_html=True)
            return
        elif kind == "final":
            ss.final_state = payload

    state = ss.final_state or {}
    final = state.get("final_answer") or "".join(tokens) or "(no answer)"
    ans_ph.markdown(final)
    step_ph.markdown(stage_strip_html(Stage.DONE, set(ss.progress["visited"]) | {Stage.DONE}),
                     unsafe_allow_html=True)

    ss.history.append({
        "role": "assistant",
        "content": final,
        "badges": state.get("final_verdict_badges", []),
        "meta": {
            "decision": state.get("final_decision"),
            "route": state.get("selected_kb"),
            "model": (state.get("model_used") or "").split("/")[-1],
            "latency_s": round((time.time() - t0), 2),
            "graph_ms": state.get("total_latency_ms"),
        },
    })


def _badge_html(b: str) -> str:
    low = b.lower()
    cls = "info"
    if "safe" in low:
        cls = "safe"
    elif "block" in low:
        cls = "block"
    elif "harm" in low:
        cls = "harm"
    elif "cache" in low:
        cls = "cache"
    return f'<span class="cp-badge {cls}">{html.escape(b)}</span>'


# ============================ TAB 1 ============================
with tab_q:
    with st.sidebar:
        st.markdown("### 🎛️ Demo prompts")
        opts = ["—"] + [f"{p['id']} · {p['prompt'][:46]}" for p in DEMO_PROMPTS]
        sel = st.selectbox("pick one", opts, label_visibility="collapsed")
        if sel != "—":
            p = DEMO_PROMPTS[opts.index(sel) - 1]
            st.caption(f"**{p['functionality']}**")
            st.info(p["prompt"])
            if st.button("▶  Send this prompt", type="primary", use_container_width=True):
                ss.queued_prompt = p["prompt"]
                st.rerun()
        st.divider()
        c1, c2 = st.columns(2)
        if c1.button("🗑️ Cache", use_container_width=True, help="clear the semantic cache"):
            get_cache().clear()
            st.toast("Semantic cache cleared")
            st.rerun()
        if c2.button("🔄 New chat", use_container_width=True):
            ss.thread_id = new_thread_id()
            ss.history, ss.final_state, ss.pending_interrupt = [], {}, None
            st.rerun()
        st.caption(f"thread `{ss.thread_id}`")

    # history
    for m in ss.history:
        with st.chat_message(m["role"], avatar="🧑‍💻" if m["role"] == "user" else "🛡️"):
            st.markdown(m["content"])
            if m.get("badges"):
                st.markdown(" ".join(_badge_html(str(b)) for b in m["badges"]), unsafe_allow_html=True)
            mt = m.get("meta") or {}
            if mt:
                bits = [x for x in [
                    mt.get("decision") and f"decision **{mt['decision']}**",
                    mt.get("route") and f"route `{mt['route']}`",
                    mt.get("model") and f"model `{mt['model']}`",
                    mt.get("latency_s") and f"⏱ {mt['latency_s']}s",
                ] if x]
                if bits:
                    st.caption("  ·  ".join(bits))

    # HITL panel
    if ss.pending_interrupt:
        pi = ss.pending_interrupt
        st.warning("🙋 **Human input required to continue** (one-shot)")
        st.markdown(f"**{pi.get('question', 'More information is needed.')}**")
        with st.expander("draft answer / why"):
            st.write(pi.get("draft_answer", ""))
            st.caption(pi.get("perf_reasoning", "") or pi.get("resp_status", ""))
        reply = st.text_input("Your answer", key="hitl_reply")
        if st.button("Continue pipeline ▶", type="primary") and reply.strip():
            ss.history.append({"role": "user", "content": f"↳ {reply}"})
            with st.chat_message("assistant", avatar="🛡️"):
                _run(None, resume=reply.strip())
            st.rerun()

    prompt = st.chat_input("Ask about refunds, HR policy, Azure, meetings, content-safety…")
    if ss.queued_prompt and not prompt:
        prompt = ss.queued_prompt
    ss.queued_prompt = None

    if prompt and not ss.pending_interrupt:
        ss.history.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(prompt)
        with st.chat_message("assistant", avatar="🛡️"):
            _run(prompt)
        st.rerun()

# ============================ TAB 2 ============================
with tab_d:
    render_dashboard(ss.final_state, ss.thread_id)

# ============================ TAB 3 ============================
with tab_w:
    render_workflow(ss.progress, ss.final_state)

# ============================ TAB 4 ============================
with tab_e:
    render_evidence(ss.final_state)
