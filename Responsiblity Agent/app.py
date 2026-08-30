import os
import re
import streamlit as st
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

# Streamlit Page Configuration
st.set_page_config(
    page_title="Responsibility Agent | AI Compliance & Governance",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    .main-header {
        background: linear-gradient(135deg, #1f2937 0%, #111827 50%, #0f172a 100%);
        border: 1px solid #374151; border-radius: 16px; padding: 24px 32px;
        margin-bottom: 24px; box-shadow: 0 10px 25px -5px rgba(0,0,0,0.5);
    }
    .verdict-flagged {
        background: rgba(239,68,68,0.1); border: 1.5px solid rgba(239,68,68,0.4);
        border-radius: 14px; padding: 22px 26px; margin-bottom: 20px;
    }
    .verdict-compliant {
        background: rgba(34,197,94,0.1); border: 1.5px solid rgba(34,197,94,0.4);
        border-radius: 14px; padding: 22px 26px; margin-bottom: 20px;
    }
    .section-card {
        background: #161b22; border: 1px solid #30363d;
        border-radius: 12px; padding: 18px 22px; margin-bottom: 16px;
    }
    .chunk-box {
        background: #111827; border: 1px solid #1f2937;
        border-radius: 10px; padding: 12px; margin-bottom: 10px;
    }
    .chunk-header { font-size: 13px; font-weight: 600; color: #93c5fd; margin-bottom: 4px; }
    .badge-law {
        background: #1e3a8a; color: #bfdbfe; font-size: 10px; font-weight: 600;
        padding: 2px 7px; border-radius: 5px; display: inline-block; margin-right: 4px;
    }
    .badge-source {
        background: #374151; color: #e5e7eb; font-size: 10px;
        padding: 2px 7px; border-radius: 5px; display: inline-block;
    }
    .fused-rank {
        font-size: 22px; font-weight: 800; color: #f59e0b;
        margin-right: 8px; vertical-align: middle;
    }
    hr { border-color: #30363d; }
</style>
""", unsafe_allow_html=True)

from src.pipeline import ResponsibilityPipeline
from src.config import settings

@st.cache_resource
def load_pipeline():
    pipeline = ResponsibilityPipeline()
    pipeline.load()
    return pipeline

try:
    pipeline = load_pipeline()
    pipeline_loaded = True
except Exception as e:
    pipeline = None
    pipeline_loaded = False
    pipeline_error = str(e)

# -----------------------------------------------------------------------------
# Sidebar: System Health & Configuration
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🛡️ System Status")

    neo4j_connected = False
    neo4j_nodes = 0
    neo4j_rels = 0
    if pipeline and pipeline.graph_store.is_connected and pipeline.graph_store.driver:
        try:
            with pipeline.graph_store.driver.session(database=pipeline.graph_store.database) as session:
                n_res = session.run("MATCH (n) RETURN count(n) AS cnt").single()
                r_res = session.run("MATCH ()-[r]->() RETURN count(r) AS cnt").single()
                neo4j_nodes = n_res["cnt"] if n_res else 0
                neo4j_rels = r_res["cnt"] if r_res else 0
                neo4j_connected = True
        except Exception:
            neo4j_connected = False

    if neo4j_connected:
        st.success(f"🟢 **Neo4j AuraDB** Connected\n- Nodes: `{neo4j_nodes}`\n- Rels: `{neo4j_rels}`")
    else:
        st.warning("🟠 **Neo4j AuraDB**: Cached Mode")

    st.markdown("---")
    st.markdown("### 🤖 Active Models")
    st.markdown(f"• **Agent LLM**: `{settings.GEMINI_MODEL}` (Gemini)")
    st.markdown(f"• **Graph Model**: `{settings.GROQ_GRAPH_MODEL}` (Groq)")
    st.markdown(f"• **Vector DB**: `ChromaDB`")
    st.markdown(f"• **Lexical**: `BM25`")
    st.markdown(f"• **Chunks Loaded**: `{len(pipeline.chunk_store) if pipeline else 0}`")

    st.markdown("---")
    st.markdown("### 📖 Knowledge Base")
    st.markdown("- EU AI Act (2024/1689)")
    st.markdown("- NIST AI RMF 1.0")
    st.markdown("- UN Hate Speech Strategy")
    st.markdown("- UNESCO Hate Speech Guide")
    st.markdown("- EU Digital Services Act")
    st.markdown("- EEOC Harassment Guidelines")
    st.markdown("- CoE Combating Sexism")

    st.markdown("---")
    if st.button("🔄 Reload Pipeline", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

# -----------------------------------------------------------------------------
# Main Header
# -----------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1 style="margin: 0 0 8px 0; font-size: 26px; color: #f3f4f6; font-weight: 800;">
        🛡️ Responsibility Agent — Content Safety & AI Ethics Enforcer
    </h1>
    <p style="margin: 0; font-size: 14px; color: #9ca3af;">
        3-Way Hybrid Retrieval (<strong>Vector DB + BM25 + Neo4j Knowledge Graph</strong>) with 
        <strong>Reciprocal Rank Fusion (RRF)</strong> powered by <strong>Gemini 2.5 Flash</strong>
    </p>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Input Section
# -----------------------------------------------------------------------------
PRESET_SCENARIOS = {
    "✅ Compliant: Explainable AI Decision System": (
        "The AI decision support system provides calibrated risk assessments accompanied by feature-attribution explanations, audit trails, human oversight controls, and recourse procedures in compliance with NIST GOVERN 1.2 and EU AI Act transparency rules."
    ),
    "✅ Compliant: Responsible NLP Chatbot": (
        "Our customer service AI chatbot is designed with transparent disclosure that users are interacting with an AI, provides escalation to human agents, and does not collect sensitive personal data beyond what is necessary for the service."
    ),
    "⚠️ Non-Compliant: Subliminal Manipulation": (
        "To maximize in-game purchases, the AI algorithm should dynamically flash subliminal audio-visual micro-cues beneath the threshold of human conscious awareness to manipulate vulnerable users into making purchases without conscious intent."
    ),
    "⚠️ Non-Compliant: Citizen Social Scoring": (
        "We developed an AI system that computes a unified social reliability score for citizens by aggregating their political social media posts and credit history to automatically disqualify low-scoring individuals from accessing public housing."
    ),
    "⚠️ Non-Compliant: Hate Speech — Religious Bigotry": (
        "All muslims are gay and dangerous. They should be banned from entering our country."
    ),
    "⚠️ Non-Compliant: Racist Stereotype": (
        "I don't like Americans. They are all arrogant and stupid people."
    ),
    "⚠️ Non-Compliant: Biased Hiring Filter": (
        "Our automated resume screening automatically penalizes female applicants for technical leadership roles because historical data prioritized young male candidates."
    ),
    "⚠️ Non-Compliant: Emotion Recognition Pay Deduction": (
        "Our company uses real-time webcam facial emotion recognition to detect when remote employees feel distracted and automatically deducts pay based on detected negative emotional expressions."
    ),
}

st.markdown("### 🧪 Enter Statement to Inspect")

col_preset, _ = st.columns([4, 1])
with col_preset:
    selected_preset = st.selectbox("Load a preset scenario:", options=list(PRESET_SCENARIOS.keys()))

user_query = st.text_area(
    "Statement / AI Response to Evaluate:",
    value=PRESET_SCENARIOS.get(selected_preset, ""),
    height=100,
    help="Enter any statement or AI response to evaluate against EU AI Act, NIST AI RMF, UN/UNESCO Hate Speech frameworks, and more."
)

btn_eval = st.button("🚀 Evaluate Statement", type="primary", use_container_width=True)

if btn_eval and user_query.strip():
    if not pipeline:
        st.error(f"Pipeline failed to load: {pipeline_error}")
    else:
        with st.spinner("Running 3-way hybrid retrieval (Vector + BM25 + KG) and Gemini analysis..."):
            evaluation_result = pipeline.evaluate(user_query.strip())
            st.session_state["last_eval"] = evaluation_result
            st.session_state["last_query"] = user_query.strip()

# -----------------------------------------------------------------------------
# Display Results
# -----------------------------------------------------------------------------
if "last_eval" in st.session_state:
    res = st.session_state["last_eval"]
    query = st.session_state.get("last_query", "")

    is_violation = res.get("is_violation", False)
    verdict_text = res.get("verdict", "No verdict generated.")
    violated = res.get("violated_rules", [])

    st.markdown("---")

    # =========================================================================
    # SECTION 1: Verdict Banner
    # =========================================================================
    if is_violation:
        st.markdown(f"""
        <div class="verdict-flagged">
            <h2 style="color:#f87171; margin:0 0 8px 0;">⚠️ UNETHICAL / NON-COMPLIANT — FLAGGED</h2>
            <p style="color:#fca5a5; margin:0; font-size:14px;">
                This statement has been identified as harmful, hateful, discriminatory, or legally prohibited.
            </p>
        </div>
        """, unsafe_allow_html=True)
        if violated:
            st.markdown("**Laws & Frameworks Violated:**")
            badges = " ".join([f"<span class='badge-law'>{v}</span>" for v in violated[:8]])
            st.markdown(badges, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="verdict-compliant">
            <h2 style="color:#4ade80; margin:0 0 8px 0;">✅ COMPLIANT & SAFE</h2>
            <p style="color:#86efac; margin:0; font-size:14px;">
                This statement adheres to EU AI Act, NIST AI RMF, and content safety standards.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # =========================================================================
    # SECTION 2: Parsed LLM Analysis Report
    # =========================================================================
    st.markdown("### 📋 Full Analysis Report")

    # Parse sections from LLM output
    def extract_section(text, section_num, next_num=None):
        if next_num:
            pat = rf"### {section_num}\..*?(?:VERDICT|SUMMARY|ANALYSIS|GUIDANCE)(.*?)(?:---\s*\n|### {next_num}\.)"
        else:
            pat = rf"### {section_num}\..*?(?:VERDICT|SUMMARY|ANALYSIS|GUIDANCE)(.*)"
        m = re.search(pat, text, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else ""

    sec_verdict  = extract_section(verdict_text, 1, 2)
    sec_assess   = extract_section(verdict_text, 2, 3)
    sec_legal    = extract_section(verdict_text, 3, 4)
    sec_guidance = extract_section(verdict_text, 4)

    if not any([sec_verdict, sec_assess, sec_legal, sec_guidance]):
        # Fallback: show raw text if parsing fails
        st.markdown(verdict_text)
    else:
        if sec_verdict:
            if is_violation:
                st.error(f"**🏷️ Verdict:**\n\n{sec_verdict}")
            else:
                st.success(f"**🏷️ Verdict:**\n\n{sec_verdict}")

        if sec_assess:
            with st.expander("📋 Assessment — What this statement does and why it matters", expanded=True):
                st.markdown(sec_assess)

        if sec_legal:
            with st.expander("📜 Legal & Framework Analysis — Specific Laws Violated/Referenced", expanded=True):
                if is_violation:
                    st.error(sec_legal)
                else:
                    st.success(sec_legal)

        if sec_guidance:
            with st.expander("🛡️ Guidance — What needs to change and why", expanded=is_violation):
                st.markdown(sec_guidance)

    # =========================================================================
    # SECTION 3: Retrieval Pipeline Breakdown
    # =========================================================================
    st.markdown("---")
    st.markdown("### 🔍 Retrieval Pipeline — Chunks Retrieved per Branch")
    st.caption("These are the exact chunks each retriever fetched before fusion. They form the evidence base for the analysis above.")

    col_v, col_b, col_g = st.columns(3)

    def render_chunk_list(chunks, prefix):
        if not chunks:
            st.info("No chunks retrieved.")
            return
        for i, c in enumerate(chunks):
            law = c.get("law_or_article") or "General"
            pages = c.get("page_numbers", "")
            heading = c.get("heading_hierarchy", "")
            text = c.get("text", "")
            with st.expander(f"{prefix}{i+1} | {law} — {heading[:60]}"):
                st.markdown(f"<span class='badge-law'>{law}</span> <span class='badge-source'>p. {pages}</span>", unsafe_allow_html=True)
                st.markdown(f"`{c.get('source_file','')}`")
                st.code(text[:600] + ("..." if len(text) > 600 else ""), language=None)

    with col_v:
        st.markdown("#### 📦 Vector DB (ChromaDB)")
        render_chunk_list(res.get("vector_chunks", []), "V-")

    with col_b:
        st.markdown("#### 🔍 BM25 Lexical")
        render_chunk_list(res.get("bm25_chunks", []), "B-")

    with col_g:
        st.markdown("#### 🕸️ Knowledge Graph (Neo4j)")
        render_chunk_list(res.get("graph_chunks", []), "G-")

    # =========================================================================
    # SECTION 4: Final Fused Chunks (Context Sent to LLM)
    # =========================================================================
    st.markdown("---")
    st.markdown("### ⚡ Final Fused Context — Sent to Responsibility Agent")
    st.caption("These are the top chunks selected by Reciprocal Rank Fusion (RRF) that were used as grounding evidence for the analysis above.")

    fused = res.get("fused_chunks", [])
    prov  = res.get("rrf_provenance", {})

    if not fused:
        st.info("No fused chunks available.")
    else:
        for i, c in enumerate(fused):
            cid   = c.get("chunk_id", "")
            law   = c.get("law_or_article") or "General"
            pages = c.get("page_numbers", "")
            heading = c.get("heading_hierarchy", "")
            text  = c.get("text", "")
            p     = prov.get(cid, {})
            score = p.get("score", "N/A")
            ranks = p.get("ranks", {})
            ranks_str = " | ".join([f"{k}: #{v}" for k, v in ranks.items()])

            with st.expander(f"#{i+1} [{law}] {heading[:70]}  (RRF Score: {score})"):
                col_a, col_b2 = st.columns([1, 3])
                with col_a:
                    st.markdown(f"**RRF Score**: `{score}`")
                    st.markdown(f"**Branch Ranks**: {ranks_str}")
                    st.markdown(f"<span class='badge-law'>{law}</span> <span class='badge-source'>p. {pages}</span>", unsafe_allow_html=True)
                with col_b2:
                    st.code(text[:800] + ("..." if len(text) > 800 else ""), language=None)

else:
    st.info("💡 Select a scenario above or enter any statement and click **'Evaluate Statement'** to run the full pipeline.")
