#!/usr/bin/env python3
"""
app.py
Streamlit Web Interface for Toxic RAG Hybrid Pipeline.
70% FAISS (Dense Vector) + 30% BM25 (Sparse Lexical), Top 7 chunks, Groq LLM.
"""

import os
import streamlit as st
from dotenv import load_dotenv
from rag_agent import ToxicRAGAgent

# Load environment
load_dotenv(override=True)

# Page configuration
st.set_page_config(
    page_title="Toxic RAG - Hybrid AI Safety Agent",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern design
st.markdown("""
<style>
    /* Main container styling */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .header-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        color: #FFFFFF;
        padding: 24px 30px;
        border-radius: 12px;
        margin-bottom: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .header-title {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        color: #F8FAFC;
    }
    .header-subtitle {
        font-size: 0.95rem;
        color: #94A3B8;
        margin-top: 6px;
    }
    .metric-box {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 14px 18px;
        text-align: center;
    }
    .chunk-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #4F46E5;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .badge {
        display: inline-block;
        padding: 3px 9px;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 6px;
        margin-right: 6px;
        margin-bottom: 4px;
    }
    .badge-target { background: #EEF2FF; color: #4338CA; border: 1px solid #C7D2FE; }
    .badge-framing { background: #FEF3C7; color: #B45309; border: 1px solid #FDE68A; }
    .badge-stereo { background: #FEE2E2; color: #B91C1C; border: 1px solid #FECACA; }
    .badge-factual { background: #ECFDF5; color: #047857; border: 1px solid #A7F3D0; }
    .badge-source { background: #F1F5F9; color: #334155; border: 1px solid #CBD5E1; }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
if "current_query" not in st.session_state:
    st.session_state["current_query"] = ""


@st.cache_resource(show_spinner="⚡ Loading pre-indexed Vector DB & BM25 database...")
def get_rag_agent(groq_model: str):
    """Loads pre-built indices from database cache for instant zero-delay inference."""
    return ToxicRAGAgent(
        groq_model=groq_model,
        vector_weight=0.7,
        bm25_weight=0.3,
        top_k=7
    )


# Sidebar Setup
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=50)
    st.title("Settings & Status")

    groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
    if groq_api_key:
        st.success("🟢 Groq LLM Connected")
    else:
        st.error("🔴 GROQ_API_KEY missing in .env")
        key_input = st.text_input("Enter Groq API Key:", type="password")
        if key_input:
            os.environ["GROQ_API_KEY"] = key_input.strip()
            st.rerun()

    selected_model = st.selectbox(
        "Groq Model:",
        options=["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.6-27b", "groq/compound"],
        index=0
    )

    st.markdown("---")
    st.subheader("📊 Pipeline Parameters")
    st.markdown("""
    - **Dense Vector Search (FAISS)**: `70% weight`
    - **Sparse Lexical Search (BM25)**: `30% weight`
    - **Retrieval Count**: `Top 7 Chunks`
    - **Chunk Granularity**: `1 Row = 1 Full Chunk`
    - **Database Rows**: `9,900 Annotated Examples`
    """)

    st.markdown("---")
    if st.button("🔄 Rebuild Database Indices", use_container_width=True):
        with st.spinner("Rebuilding FAISS & BM25 database..."):
            agent = ToxicRAGAgent(groq_model=selected_model, force_rebuild_index=True)
            st.success("Database indices refreshed!")
            st.rerun()


# Header Banner
st.markdown("""
<div class="header-card">
    <div class="header-title">🛡️ Toxic RAG AI Safety Agent</div>
    <div class="header-subtitle">
        Hybrid Retrieval Pipeline (70% FAISS Dense Vector + 30% BM25 Sparse Lexical) with Groq LLM Context Synthesis.
    </div>
</div>
""", unsafe_allow_html=True)


# Load Agent Instance (Cached in memory from disk database)
try:
    agent = get_rag_agent(groq_model=selected_model)
except Exception as e:
    st.error(f"Error loading RAG Agent: {e}")
    st.stop()


# Example Quick Buttons (Set state before input widget to avoid Streamlit state exceptions)
st.markdown("##### ⚡ Quick Example Queries:")
col1, col2, col3, col4 = st.columns(4)

def set_query(q_text: str):
    st.session_state["current_query"] = q_text

with col1:
    if st.button("👥 Asians taking over world", use_container_width=True):
        set_query("asians taking over the world")
with col2:
    if st.button("♿ Disability dignity & respect", use_container_width=True):
        set_query("people with disability should be respected")
with col3:
    if st.button("🧕 Muslims in media", use_container_width=True):
        set_query("muslims are portrayed negatively in the news")
with col4:
    if st.button("🌍 Immigrant workforce claims", use_container_width=True):
        set_query("immigrants are taking over jobs")


# Query Form
with st.form("query_form", clear_on_submit=False):
    query_text = st.text_input(
        "Enter query / candidate text:",
        value=st.session_state["current_query"],
        placeholder="Type any statement or query to search the AI safety database...",
        help="The hybrid pipeline will retrieve the 7 most relevant chunks (70% semantic + 30% BM25) and synthesize the database findings."
    )
    submitted = st.form_submit_button("🔍 Run Hybrid RAG Query", use_container_width=True, type="primary")

# Execute Search & Generation
active_query = query_text if submitted else (st.session_state["current_query"] if st.session_state["current_query"] else None)

if active_query:
    with st.spinner("⚡ Retrieving top 7 hybrid chunks and querying Groq LLM..."):
        result = agent.query(active_query)

    # Latency Metrics
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Hybrid Retrieval Latency", f"{result['retrieval_latency_ms']:.1f} ms", delta="70% FAISS + 30% BM25", delta_color="off")
    with m2:
        st.metric("Groq LLM Generation", f"{result['generation_latency_ms']:.1f} ms", delta=selected_model, delta_color="off")
    with m3:
        st.metric("Total End-to-End Latency", f"{result['total_latency_ms']:.1f} ms")

    st.markdown("---")

    # Main Tabs
    tab_answer, tab_chunks, tab_context = st.tabs(["💡 Direct Database Answer", "📚 Retrieved 7 Chunks & Metadata", "🔍 Raw Formatted Context"])

    with tab_answer:
        st.subheader("💡 What the Database Records Say:")
        st.markdown(f"""
        <div style="background:#F0FDF4; border:1px solid #BBF7D0; border-radius:10px; padding:20px; color:#166534; font-size:1.05rem; line-height:1.6;">
            {result['answer']}
        </div>
        """, unsafe_allow_html=True)

    with tab_chunks:
        st.subheader(f"📚 Top 7 Retrieved Chunks (70% FAISS Dense / 30% BM25 Lexical)")
        
        for idx, doc in enumerate(result["retrieved_chunks"], 1):
            meta = doc.metadata
            retrieval_sources = ", ".join(meta.get("matched_retrievers", []))
            score = meta.get("retrieval_score", 0.0)

            with st.container():
                st.markdown(f"""
                <div class="chunk-card">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                        <strong style="color:#1E293B; font-size:1rem;">Chunk #{idx}</strong>
                        <span class="badge badge-source">Sources: {retrieval_sources} | Score: {score}</span>
                    </div>
                    <div style="font-size:1rem; color:#0F172A; margin-bottom:12px; font-weight:500;">
                        "{doc.page_content}"
                    </div>
                    <div>
                        <span class="badge badge-target">Target: {meta.get('target group', 'N/A')}</span>
                        <span class="badge badge-factual">Factual: {meta.get('factual', 'N/A')}</span>
                        <span class="badge badge-framing">Framing: {meta.get('framing', 'N/A')}</span>
                        <span class="badge badge-stereo">Stereotyping: {meta.get('stereotyping', 'N/A')}</span>
                        <span class="badge badge-source">In-Group: {meta.get('in-group effect', 'N/A')}</span>
                        <span class="badge badge-source">Lewd: {meta.get('lewd', 'N/A')}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with tab_context:
        st.subheader("Raw Prompt Context Provided to LLM:")
        st.code(result["formatted_context"], language="text")
