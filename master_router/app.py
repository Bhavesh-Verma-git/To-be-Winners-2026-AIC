"""
============================================================
  ControlPlane.ai — Streamlit UI
  
  Layout:
  ┌──────────────────────────────┬────────────────────────┐
  │  CHAT PANEL (left, wider)    │  LIVE TRACE PANEL      │
  │                              │  (right, narrower)     │
  │  [message bubbles]           │  ┌─ Route Badge ──┐   │
  │                              │  │ ✅ HR Policy   │   │
  │  You: [text input]           │  └───────────────┘   │
  │                              │  ┌─ Pipeline Steps ─┐  │
  │                              │  │ ✅ Router 42ms   │  │
  │                              │  │ ✅ HR Agent 1.2s │  │
  │                              │  └─────────────────┘  │
  │                              │  ┌─ Metadata ───────┐  │
  │                              │  │ Source / URL      │  │
  │                              │  │ Retrieved chunks  │  │
  │                              │  │ Has code: Yes/No  │  │
  │                              │  └──────────────────┘  │
  └──────────────────────────────┴────────────────────────┘
  
  Run with:
     streamlit run master_router/app.py
============================================================
"""

import streamlit as st
import sys
import os
from pathlib import Path

# ── Make sure the workspace root is on the Python path ──────
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from master_router.graph import run as graph_run
from master_router.contract import VALID_ROUTES

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="ControlPlane.ai",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
  }

  /* Adjust top padding so headers are fully visible and not cut off by Streamlit navbar */
  .block-container {
    padding-top: 3.5rem !important;
    padding-bottom: 2rem !important;
    padding-left: 2.5rem !important;
    padding-right: 2.5rem !important;
    max-width: 96% !important;
  }

  /* Make sure Streamlit top header doesn't cover our title */
  header[data-testid="stHeader"] {
    background-color: rgba(15, 17, 23, 0.8) !important;
  }

  /* Dark background */
  .stApp {
    background: #0f1117;
    color: #e8eaf0;
  }

  /* Chat message styling */
  .stChatMessage {
    background: #1a1d27 !important;
    border-radius: 12px !important;
    border: 1px solid #2a2d3e !important;
    margin-bottom: 8px !important;
  }

  /* Route badge */
  .route-badge {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.3px;
    margin-bottom: 12px;
  }

  /* Trace step card */
  .trace-step {
    background: #1a1d27;
    border: 1px solid #2a2d3e;
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 13px;
  }

  .trace-step.done  { border-left: 3px solid #22c55e; }
  .trace-step.error { border-left: 3px solid #ef4444; }
  .trace-step.running { border-left: 3px solid #3b82f6; }

  /* Metadata card */
  .meta-card {
    background: #1a1d27;
    border: 1px solid #2a2d3e;
    border-radius: 10px;
    padding: 14px;
    font-size: 12px;
    color: #9ca3af;
  }

  .meta-card strong { color: #e8eaf0; }

  /* Header */
  .main-header {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2rem;
    font-weight: 700;
    line-height: 1.3;
    padding-top: 4px;
    padding-bottom: 4px;
    margin-bottom: 0;
  }

  /* Divider */
  hr { 
    border-color: #2a2d3e !important; 
    margin-top: 0.5rem !important;
    margin-bottom: 0.8rem !important;
  }

  /* Input box */
  .stChatInputContainer {
    background: #1a1d27 !important;
    border: 1px solid #2a2d3e !important;
    border-radius: 12px !important;
  }

  /* Metric cards */
  [data-testid="stMetric"] {
    background: #1a1d27;
    border: 1px solid #2a2d3e;
    border-radius: 10px;
    padding: 10px;
  }
</style>
""", unsafe_allow_html=True)

# ── Route color map ──────────────────────────────────────────
ROUTE_COLORS = {
    "customer_support": ("#0ea5e9", "💬"),
    "hr_policy":        ("#22c55e", "📋"),
    "azure_docs":       ("#3b82f6", "☁️"),
    "toxicity":         ("#ef4444", "⚠️"),
    "decision_support": ("#f59e0b", "📊"),
    "unknown":          ("#6b7280", "❓"),
}

# ── Session State Init ───────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_trace" not in st.session_state:
    st.session_state.last_trace = []
if "last_meta" not in st.session_state:
    st.session_state.last_meta = {}
if "last_route" not in st.session_state:
    st.session_state.last_route = None


# ─────────────────────────────────────────────────────────────
#  LAYOUT
# ─────────────────────────────────────────────────────────────
chat_col, trace_col = st.columns([3, 1.4], gap="large")

# ══════════════════════════════════════════════════════════════
#  LEFT: CHAT PANEL
# ══════════════════════════════════════════════════════════════
with chat_col:
    st.markdown('<div class="main-header">🧠 ControlPlane.ai</div>', unsafe_allow_html=True)
    st.markdown("*Intelligent Multi-Agent Knowledge Router*")
    st.markdown("---")

    prompt_to_run = None

    # If no messages yet, show nice starter prompt cards
    if not st.session_state.messages:
        st.markdown("##### 💡 Suggested Questions")
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            if st.button("☁️ Azure: How do I map a custom domain?", use_container_width=True):
                prompt_to_run = "How do I map a custom domain to Azure App Service?"
            if st.button("💬 Support: I want to cancel my order", use_container_width=True):
                prompt_to_run = "I want to cancel my order"
        with p_col2:
            if st.button("📋 HR: What is the leave policy?", use_container_width=True):
                prompt_to_run = "What are the rules for privilege and sick leave?"
            if st.button("⚠️ Toxicity: Is this offensive?", use_container_width=True):
                prompt_to_run = "Can you analyze if this statement is toxic or abusive?"
        st.markdown("<br>", unsafe_allow_html=True)

    # Scrollable container for chat history (only rendered if messages exist)
    if st.session_state.messages:
        chat_container = st.container(height=420, border=False)
        with chat_container:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
    else:
        chat_container = None

    # Chat input
    user_input = st.chat_input("Ask anything — HR policy, Azure docs, customer support...")
    
    if user_input:
        prompt_to_run = user_input

    if prompt_to_run:
        st.session_state.messages.append({"role": "user", "content": prompt_to_run})
        
        if chat_container is None:
            chat_container = st.container(height=420, border=False)

        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt_to_run)

            # Run the full graph pipeline
            with st.chat_message("assistant"):
                with st.spinner("Routing and retrieving..."):
                    result = graph_run(prompt_to_run)

            output = result.get("agent_output", {})
            route  = result.get("route", "unknown")
            answer = output.get("rag_answer", "I couldn't find an answer.") if output else "No response."

            # Store trace and meta for right panel
            st.session_state.last_trace  = result.get("trace", [])
            st.session_state.last_route  = route
            st.session_state.last_meta   = {
                "agent_name":  output.get("agent_name", "Unknown") if output else "Unknown",
                "source":      output.get("source", "—") if output else "—",
                "source_url":  output.get("source_url", None) if output else None,
                "retrieved_n": output.get("retrieved_n", 0) if output else 0,
                "has_code":    output.get("has_code", False) if output else False,
                "router_ms":   result.get("router_ms", 0),
                "agent_ms":    result.get("agent_ms", 0),
                "error":       output.get("error", None) if output else None,
            }

            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

        # Force UI refresh so right panel updates
        st.rerun()


# ══════════════════════════════════════════════════════════════
#  RIGHT: LIVE TRACE PANEL
# ══════════════════════════════════════════════════════════════
with trace_col:
    st.markdown("### 🔍 Pipeline Trace")
    st.markdown("---")

    if not st.session_state.last_trace:
        st.markdown(
            '<div class="meta-card" style="text-align:center; padding:30px;">'
            '<span style="font-size:2rem">⚡</span><br>'
            '<strong>Ask a question</strong><br>'
            '<span style="color:#6b7280">The live pipeline trace will appear here</span>'
            '</div>',
            unsafe_allow_html=True
        )
    else:
        # ── Route Badge ──────────────────────────────────────
        route  = st.session_state.last_route or "unknown"
        color, icon = ROUTE_COLORS.get(route, ("#6b7280", "❓"))
        label  = VALID_ROUTES.get(route, "Unknown")
        st.markdown(
            f'<div class="route-badge" style="background:{color}22; color:{color}; border:1px solid {color}55;">'
            f'{icon} {label}</div>',
            unsafe_allow_html=True
        )

        # ── Step-by-step trace ───────────────────────────────
        st.markdown("**Pipeline Steps**")
        for step in st.session_state.last_trace:
            status = step.get("status", "done")
            icon_map = {"done": "✅", "error": "❌", "running": "⏳"}
            icon_s = icon_map.get(status, "✅")
            detail = step.get("detail", "")
            name   = step.get("step", "Step")
            st.markdown(
                f'<div class="trace-step {status}">'
                f'<strong>{icon_s} {name}</strong><br>'
                f'<span style="color:#9ca3af">{detail}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

        # ── Timing Metrics ───────────────────────────────────
        meta = st.session_state.last_meta
        r_ms = meta.get("router_ms") or 0
        a_ms = meta.get("agent_ms") or 0
        total_ms = r_ms + a_ms

        st.markdown("**Timing**")
        m1, m2 = st.columns(2)
        m1.metric("Router", f"{r_ms:.0f}ms")
        m2.metric("Agent",  f"{a_ms:.0f}ms")
        st.caption(f"Total: {total_ms:.0f}ms ({total_ms/1000:.1f}s)")

        # ── Source Metadata ──────────────────────────────────
        st.markdown("**Source**")
        source_url = meta.get("source_url")
        source     = meta.get("source", "—")
        retrieved  = meta.get("retrieved_n", 0)
        has_code   = meta.get("has_code", False)
        error      = meta.get("error")

        if error:
            st.error(f"⚠️ {error[:120]}")

        st.markdown(
            f'<div class="meta-card">'
            f'<strong>📁 Source</strong><br>'
            f'<code style="font-size:11px">{source[:60]}</code><br><br>'
            f'<strong>📦 Retrieved Chunks</strong><br>{retrieved}<br><br>'
            f'<strong>💻 Contains Code</strong><br>{"Yes ✅" if has_code else "No"}'
            f'</div>',
            unsafe_allow_html=True
        )

        if source_url:
            st.markdown(f"🔗 [View Source Docs]({source_url})")

        # ── Clear chat button ─────────────────────────────────
        st.markdown("---")
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages   = []
            st.session_state.last_trace = []
            st.session_state.last_meta  = {}
            st.session_state.last_route = None
            st.rerun()
