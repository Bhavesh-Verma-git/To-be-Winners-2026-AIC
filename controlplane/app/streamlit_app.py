"""
ControlPlane.ai - Streamlit UI (4 tabs).

    Tab 1  Query / Answer        - ask, watch the "mulling" ticker + pipeline steps + tokens
                                   stream, see the VERDICT (SAFE / BLOCK / EDIT / HITL), resolve HITL
    Tab 2  Dashboard             - node & model latency, cost (Gemini only), chunks, scores
    Tab 3  Live Workflow         - the pipeline graph with the path actually taken
    Tab 4  Retrieval & Evidence  - every retrieved chunk + the laws a harmful query violated

Run:  streamlit run controlplane/app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import html
import time

import streamlit as st

from controlplane.app.mulling import MullFeed, mull_html
from controlplane.app.runner import new_thread_id, stream_run
from controlplane.app.tab_dashboard import render_dashboard
from controlplane.app.tab_evidence import render_evidence
from controlplane.app.tab_workflow import STAGE_STEPS, render_workflow, stage_strip_html
from controlplane.cache import get_cache
from controlplane.config import KB_IDS, KB_LABELS, settings
from controlplane.prompts import DEMO_PROMPTS
from controlplane.state import Stage

_AUTO = "Auto — router decides"

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
  .cp-mull { font-size:12.5px; color:#8a9ec9; font-style:italic; line-height:1.7;
     padding:6px 2px; letter-spacing:.2px; }
  .cp-mull-dot { color:#7aa2ff; animation:cp-pulse 1s ease-in-out infinite; font-style:normal; }
  @keyframes cp-pulse { 0%,100%{opacity:.25} 50%{opacity:1} }
  .cp-verdict { display:inline-block; padding:6px 14px; border-radius:10px; font-weight:800;
     font-size:13px; letter-spacing:.5px; margin:2px 0 8px; }
  .cp-verdict.safe  { background:#0f2a22; color:#57e0c7; border:1px solid #1f6b52; }
  .cp-verdict.block { background:#2a1414; color:#ff8f8f; border:1px solid #6b2a2a; }
  .cp-verdict.edit  { background:#1c2340; color:#b9a4ff; border:1px solid #4a3d8a; }
  .cp-verdict.hitl  { background:#2a2413; color:#ffd88a; border:1px solid #7a6320; }
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
ss.setdefault("forced_kb", None)          # None -> Auto routing; else a KB id (Mode 2)


# ---------------------------------------------------------------- warm start ----
# Load every embedder / index / classifier ONCE per server process (not per
# session, not per query). Without this the first visitor after a cold start
# pays the full ~60-90s model load inside their first query and it looks like a
# hang. st.cache_resource makes the body run exactly once even with concurrent
# first visitors. Set CP_SKIP_WARMUP=1 to opt out.
@st.cache_resource(show_spinner=False)
def _warm_pipeline() -> bool:
    import os as _os

    if _os.getenv("CP_SKIP_WARMUP", "").lower() in {"1", "true", "yes"}:
        return True
    try:
        from controlplane.retrievers.registry import get_minilm, warm_all

        get_minilm()
        warm_all(verbose=False)
    except Exception:
        pass
    for _mod, _fn in (
        ("controlplane.performance.xgboost_infer", "warmup"),
        ("controlplane.performance.entity_drift", "warmup"),
    ):
        try:
            getattr(__import__(_mod, fromlist=[_fn]), _fn)()
        except Exception:
            pass
    try:
        from controlplane.responsibility import get_responsibility_kb, get_toxicity_ensemble

        get_responsibility_kb()
        get_toxicity_ensemble().score_sync("warm up text")
    except Exception:
        pass
    return True


with st.spinner("⚙️  First launch — loading models & indexes (~60–90s, one time)…"):
    _warm_pipeline()


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
    mull_ph = st.empty()
    verdict_ph = st.empty()
    ans_ph = st.empty()
    tokens: list[str] = []
    t0 = time.time()
    feed = MullFeed()

    def _paint_steps():
        step_ph.markdown(stage_strip_html(ss.progress["stage"], set(ss.progress["visited"])),
                         unsafe_allow_html=True)

    def _paint_mull():
        feed.add_stage(ss.progress["stage"])
        mull_ph.markdown(mull_html(feed.pump()), unsafe_allow_html=True)

    _paint_steps()
    _paint_mull()
    _last_paint = 0.0
    _streamed = ""            # authoritative answer text (from answer_done if present)
    for kind, payload in stream_run(prompt, ss.thread_id, resume=resume, progress=ss.progress,
                                    forced_kb=ss.forced_kb):
        if kind == "node":
            _paint_steps()
            _paint_mull()
        elif kind == "answer_start":
            ans_ph.markdown(" ▌")             # cursor shows immediately, before first-token latency
        elif kind == "reset":
            tokens = []                       # a failed model attempt - clear its partial output
            ans_ph.markdown("_regenerating…_")
        elif kind == "token":
            tokens.append(payload)
            now = time.time()
            if now - _last_paint > 0.03:      # ~33 fps repaint
                ans_ph.markdown("".join(tokens) + " ▌")
                _last_paint = now
        elif kind == "answer_done":
            _streamed = payload or "".join(tokens)
            ans_ph.markdown(_streamed)        # snap to the complete text (fills any streaming gap)
        elif kind == "interrupt":
            ss.pending_interrupt = payload
            mull_ph.empty()
            step_ph.markdown(stage_strip_html(Stage.HITL, set(ss.progress["visited"])),
                             unsafe_allow_html=True)
            verdict_ph.markdown(_verdict_html("HUMAN-IN-THE-LOOP"), unsafe_allow_html=True)
            return
        elif kind == "final":
            ss.final_state = payload

    state = ss.final_state or {}
    final = state.get("final_answer") or _streamed or "".join(tokens) or "(no answer)"
    mull_ph.empty()
    verdict_ph.markdown(_verdict_html(state.get("final_verdict")), unsafe_allow_html=True)
    ans_ph.markdown(final)
    step_ph.markdown(stage_strip_html(Stage.DONE, set(ss.progress["visited"]) | {Stage.DONE}),
                     unsafe_allow_html=True)

    ss.history.append({
        "role": "assistant",
        "content": final,
        "verdict": state.get("final_verdict"),
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


def _verdict_html(v: str | None) -> str:
    if not v:
        return ""
    low = v.lower()
    cls = "safe"
    if "block" in low:
        cls = "block"
    elif "edit" in low:
        cls = "edit"
    elif "human" in low or "hitl" in low:
        cls = "hitl"
    return f'<span class="cp-verdict {cls}">VERDICT: {html.escape(v.upper())}</span>'


# ============================ TAB 1 ============================
with tab_q:
    with st.sidebar:
        st.markdown("### 🎛️ Knowledge base")
        _kb_opts = [_AUTO] + [f"{KB_LABELS.get(k, k)}  ({k})" for k in KB_IDS]
        _kb_sel = st.selectbox(
            "retriever", _kb_opts, label_visibility="collapsed",
            help="Auto = the intelligent router picks the KB. Pick one to force it "
                 "(Mode 2: that retriever is used directly, no routing).",
        )
        ss.forced_kb = None if _kb_sel == _AUTO else KB_IDS[_kb_opts.index(_kb_sel) - 1]
        if ss.forced_kb:
            st.caption(f"🔒 forced → `{ss.forced_kb}` (router bypassed)")
        else:
            st.caption("🧭 auto routing (LLM agent + semantic fallback)")
        st.divider()

        st.markdown("### 🎛️ Demo prompts")
        # follow the knowledge-base selector: Auto -> every prompt; a forced KB ->
        # only the prompts that exercise that KB, so the two controls stay in sync.
        if ss.forced_kb:
            _pool = [p for p in DEMO_PROMPTS if p.get("kb") == ss.forced_kb] or list(DEMO_PROMPTS)
            st.caption(f"showing **{len(_pool)}** prompts for `{ss.forced_kb}`")
        else:
            _pool = list(DEMO_PROMPTS)
            st.caption(f"showing **all {len(_pool)}** prompts (Auto routing)")
        opts = ["—"] + [f"{p['id']} · {p['prompt'][:46]}" for p in _pool]
        sel = st.selectbox("pick one", opts, label_visibility="collapsed",
                           key=f"demo_sel_{ss.forced_kb or 'auto'}")
        if sel != "—":
            p = _pool[opts.index(sel) - 1]
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
            if m.get("verdict"):
                st.markdown(_verdict_html(m["verdict"]), unsafe_allow_html=True)
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
        st.markdown(_verdict_html("HUMAN-IN-THE-LOOP"), unsafe_allow_html=True)
        st.warning("🙋 **The pipeline paused — it needs one detail from you to answer this.**")
        st.markdown(pi.get("question", "More information is needed."))
        cols = st.columns(2)
        cols[0].metric("routed to", pi.get("selected_kb", "—"))
        cols[1].metric("performance", pi.get("perf_verdict", "—"))
        with st.expander("what the pipeline had so far"):
            st.caption(f"draft: {pi.get('draft_answer', '') or '—'}")
            st.caption(pi.get("perf_reasoning", "") or pi.get("resp_status", ""))
        reply = st.text_area("Your answer / clarification", key="hitl_reply",
                             placeholder="e.g. casual leave — how many days per year")
        c1, c2 = st.columns([1, 3])
        if c1.button("Submit & re-run ▶", type="primary"):
            if reply.strip():
                ss.history.append({"role": "user", "content": f"↳ (clarification) {reply.strip()}"})
                ss.pending_interrupt = None
                with st.chat_message("assistant", avatar="🛡️"):
                    _run(None, resume=reply.strip())
                st.rerun()
            else:
                st.error("Type your answer first, then Submit.")
        c2.caption("Your text is **merged into the question** and the **whole pipeline re-runs "
                   "from the start** (guardrails → cache → routing → retrieval → answer → checks).")

    prompt = st.chat_input("Ask about refunds, HR policy, Azure, meetings, content-safety…")
    if ss.queued_prompt and not prompt:
        prompt = ss.queued_prompt
    ss.queued_prompt = None

    if prompt:
        # If a HITL request is still open and the user typed a NEW query in the main
        # box (instead of answering in the HITL panel), treat it as a fresh query:
        # drop the stale interrupt + its paused checkpoint so it can't contaminate
        # the new run.
        if ss.pending_interrupt:
            ss.pending_interrupt = None
            ss.thread_id = new_thread_id()
            ss.final_state = {}
            st.toast("Started a new query — the pending clarification request was dismissed.")
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
