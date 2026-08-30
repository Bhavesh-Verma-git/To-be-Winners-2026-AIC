import os
# Suppress tokenizers fork warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from retriever import HybridRetriever
from agent import DecisionSupportAgent

# Load environment
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Customer Decision Support RAG",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, premium aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        border: 1px solid #334155;
        margin-bottom: 1.5rem;
    }
    
    .metric-badge {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .badge-total { background-color: #3b82f622; color: #60a5fa; border: 1px solid #3b82f655; }
    .badge-retrieval { background-color: #10b98122; color: #34d399; border: 1px solid #10b98155; }
    .badge-llm { background-color: #8b5cf622; color: #a78bfa; border: 1px solid #8b5cf655; }
    .badge-model { background-color: #f59e0b22; color: #fbbf24; border: 1px solid #f59e0b55; }
    
    .answer-card {
        background-color: #0f172a;
        border: 1px solid #3b82f6;
        border-left: 5px solid #3b82f6;
        border-radius: 10px;
        padding: 1.25rem 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
    }
    
    .parent-chunk-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

CURATED_QUESTIONS = [
    "What was discussed in the meeting regarding including an LCD screen on the remote control?",
    "What target demographic age group was decided for the remote control and why?",
    "What were the arguments for and against using solar power versus standard batteries?",
    "What materials (plastic vs. rubber/titanium) were proposed for the remote's body and grip?",
    "What shape was chosen for the remote control (curved/banana shape vs. flat/rectangular)?",
    "What was decided about the number and placement of buttons on the remote?",
    "What were the opinions on replacing arrow keys with a scroll wheel or joystick?",
    "How did the team evaluate adding speech recognition or voice control features?",
    "What feature was suggested to help users find a lost remote control in the room?",
    "What is the maximum target production cost and selling price for the remote control?",
    "What colors and corporate branding requirements were specified by the company?",
    "How did the team decide to handle advanced TV functions like Menu and Teletext?",
    "Was a docking station or recharging cradle considered for the remote control?",
    "What did the usability survey of 100 participants reveal about existing remote controls?",
    "Did the team discuss illuminated buttons or glow-in-the-dark features for dark rooms?",
    "What was discussed regarding making the remote waterproof or splash-resistant?",
    "How did the Project Manager allocate design responsibilities among the team members?",
    "What trade-offs were made between incorporating advanced features and staying within the cost budget?"
]

@st.cache_resource(show_spinner="Initializing FAISS Vector Store and BM25 Retriever...")
def get_retriever():
    retriever = HybridRetriever()
    retriever.load()
    return retriever

# Initialize resources
retriever = get_retriever()
agent = DecisionSupportAgent()

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/decision.png", width=64)
    st.title("⚙️ RAG Settings")
    
    st.markdown("### 🔑 Groq API Configuration")
    env_api_key = os.getenv("GROQ_API_KEY", "")
    api_key_input = st.text_input(
        "Groq API Key",
        value=env_api_key if env_api_key != "your_groq_api_key_here" else "",
        type="password",
        help="Enter your Groq API Key from console.groq.com. Stored in memory or .env."
    )
    
    env_default_model = os.getenv("GROQ_MODEL", "qwen/qwen3.6-27b")
    model_options = [
        "qwen/qwen3.6-27b",
        "qwen-2.5-32b",
        "qwen/qwen-2.5-72b-instruct",
        "qwen-qwq-32b",
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "deepseek-r1-distill-llama-70b",
        "gemma2-9b-it",
        "Custom Model..."
    ]
    
    default_idx = model_options.index(env_default_model) if env_default_model in model_options else 0
    selected_model = st.selectbox(
        "Groq Model",
        options=model_options,
        index=default_idx,
        help="Select ultra-low latency model hosted on Groq."
    )
    
    if selected_model == "Custom Model...":
        active_model = st.text_input("Enter Model ID", value="qwen/qwen3.6-27b")
    else:
        active_model = selected_model
        
    st.markdown("---")
    st.markdown("### 🔍 Hybrid Retrieval Tuning")
    
    top_k = st.slider("Top-K Parents Retrieved", min_value=1, max_value=10, value=5, step=1,
                      help="Number of distinct parent chunks resolved and supplied as context.")
    
    dense_weight = st.slider("Dense FAISS Weight", min_value=0.0, max_value=1.0, value=0.5, step=0.05,
                             help="Weight given to FAISS dense semantic embeddings.")
    sparse_weight = 1.0 - dense_weight
    st.caption(f"⚖️ Hybrid Mix: **{dense_weight*100:.0f}% Dense FAISS** + **{sparse_weight*100:.0f}% Sparse BM25**")
    
    st.markdown("---")
    st.markdown("### 📊 Index Statistics")
    if retriever.is_loaded:
        st.write(f"• **Meetings Indexed:** {len(set([p['meeting_id'] for p in retriever.parents_store.values()]))}")
        st.write(f"• **Parent Chunks:** {len(retriever.parents_store):,}")
        st.write(f"• **Child Chunks:** {len(retriever.children_store):,}")
        st.write(f"• **Vector Dimension:** 384 (`all-MiniLM-L6-v2`)")
        st.write(f"• **Hybrid Retriever:** FAISS FlatIP + BM25Okapi")

# ----------------- MAIN CONTENT -----------------
st.markdown("""
<div class="main-header">
    <h2 style="margin:0; color:#f8fafc;">🎯 Customer Decision Support RAG</h2>
    <p style="margin:0.4rem 0 0 0; color:#94a3b8; font-size: 0.95rem;">
        Low-Latency Meeting Decision Intelligence with <b>Speaker-Aware Parent-Child Chunking</b>, <b>FAISS + BM25 Hybrid Retrieval</b>, and <b>Groq LLM</b>.
    </p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab_rag, tab_explorer, tab_dataset = st.tabs(["🚀 Decision Support Q&A", "🧩 Parent-Child Explorer", "📁 Dataset Overview"])

with tab_rag:
    # Curated Question Selector
    st.markdown("#### 💡 Curated Dataset Questions (18 Domain-Specific Questions)")
    
    col_q1, col_q2 = st.columns([3, 1])
    with col_q1:
        dropdown_selection = st.selectbox(
            "Select a curated question from the dataset:",
            options=["-- Select a question --"] + CURATED_QUESTIONS,
            index=0
        )
    with col_q2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("🎲 Random Sample Question", use_container_width=True):
            import random
            st.session_state["query_input"] = random.choice(CURATED_QUESTIONS)
            st.rerun()

    # Quick topic pills
    st.markdown("**Quick Topics:**")
    pill_cols = st.columns(6)
    if pill_cols[0].button("📺 LCD Screen"):
        st.session_state["query_input"] = "What was discussed in the meeting regarding including an LCD screen on the remote control?"
        st.rerun()
    if pill_cols[1].button("⚡ Solar vs Battery"):
        st.session_state["query_input"] = "What were the arguments for and against using solar power versus standard batteries?"
        st.rerun()
    if pill_cols[2].button("🎯 Target Group"):
        st.session_state["query_input"] = "What target demographic age group was decided for the remote control and why?"
        st.rerun()
    if pill_cols[3].button("🛡️ Casing Material"):
        st.session_state["query_input"] = "What materials (plastic vs. rubber/titanium) were proposed for the remote's body and grip?"
        st.rerun()
    if pill_cols[4].button("📍 Remote Locator"):
        st.session_state["query_input"] = "What feature was suggested to help users find a lost remote control in the room?"
        st.rerun()
    if pill_cols[5].button("💰 Budget & Cost"):
        st.session_state["query_input"] = "What is the maximum target production cost and selling price for the remote control?"
        st.rerun()

    # Determine default text
    default_query = ""
    if dropdown_selection != "-- Select a question --":
        default_query = dropdown_selection
    elif "query_input" in st.session_state:
        default_query = st.session_state["query_input"]

    # Query Input
    query_text = st.text_area(
        "**Enter or edit your question:**",
        value=default_query,
        placeholder="e.g., What was discussed in the meeting regarding LCD screen or casing materials?",
        height=70
    )

    submit_col1, submit_col2, _ = st.columns([1.5, 1.5, 5])
    with submit_col1:
        run_query = st.button("⚡ Get Decision Answer", type="primary", use_container_width=True)
    with submit_col2:
        clear_btn = st.button("🗑️ Clear", use_container_width=True)
        if clear_btn:
            st.session_state.pop("query_input", None)
            st.rerun()

    if run_query and query_text.strip():
        with st.spinner("Searching hybrid indices & querying Groq LLM..."):
            response = agent.answer_query(
                query=query_text.strip(),
                retriever=retriever,
                top_k=top_k,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight,
                model_name=active_model,
                api_key=api_key_input
            )

        st.markdown("### 💬 Direct Decision Answer")
        
        # Latency Badges
        st.markdown(f"""
        <div>
            <span class="metric-badge badge-total">⚡ Total: {response['total_latency_ms']} ms</span>
            <span class="metric-badge badge-retrieval">🔍 Retrieval: {response['retrieval_latency_ms']} ms</span>
            <span class="metric-badge badge-llm">🤖 LLM: {response['llm_latency_ms']} ms</span>
            <span class="metric-badge badge-model">🏷️ Model: {response['model_used']}</span>
            <span class="metric-badge badge-retrieval">🎯 Top-K Parents: {len(response['results'])}</span>
        </div>
        """, unsafe_allow_html=True)

        # Answer Box
        st.markdown(f"""
        <div class="answer-card">
            {response['answer']}
        </div>
        """, unsafe_allow_html=True)

        # Inspection Accordion
        with st.expander(f"🔍 Context & Retrieval Inspector ({len(response['results'])} Resolved Parent Chunks)", expanded=True):
            tab_context, tab_scores = st.tabs(["📑 Resolved Parent Chunks (LLM Context)", "📊 Hybrid Scoring Breakdown"])
            
            with tab_context:
                for idx, res in enumerate(response["results"]):
                    st.markdown(f"""
                    <div class="parent-chunk-card">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 0.5rem;">
                            <span style="font-weight: 700; color: #60a5fa;">Chunk #{idx+1} | Meeting ID: {res['meeting_id']} | {res['speaker']} (Parent ID: <code>{res['parent_id']}</code>)</span>
                            <span style="font-size: 0.85rem; color: #34d399; font-weight: 600;">Hybrid Score: {res['hybrid_score']:.4f} (FAISS: {res['dense_score']:.3f} | BM25: {res['bm25_score']:.3f})</span>
                        </div>
                        <div style="font-size: 0.85rem; color: #94a3b8; margin-bottom: 0.4rem;">
                            <b>Matched Child Fragment:</b> <i>"{res['matched_child']}"</i>
                        </div>
                        <div style="background-color: #0f172a; padding: 0.75rem; border-radius: 6px; font-size: 0.9rem; line-height: 1.5; color: #f1f5f9; border-left: 3px solid #10b981;">
                            <b>Dialogue Chunk Data (Context to LLM):</b><br>
                            {res['parent_text']}
                        </div>
                        <div style="margin-top: 0.5rem; font-size: 0.8rem; color: #cbd5e1; background-color: #1e1e2f; padding: 0.5rem; border-radius: 4px;">
                            <b>Executive Summary (Metadata):</b> {res['metadata'].get('summary', 'N/A')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
            with tab_scores:
                score_data = []
                for i, r in enumerate(response["results"]):
                    score_data.append({
                        "Rank": i + 1,
                        "Parent ID": r["parent_id"],
                        "Meeting ID": r["meeting_id"],
                        "Speaker": r["speaker"],
                        "Hybrid Score (0.5/0.5)": round(r["hybrid_score"], 4),
                        "FAISS Dense Score": round(r["dense_score"], 4),
                        "BM25 Sparse Score": round(r["bm25_score"], 4),
                        "Matched Child ID": r["child_id"]
                    })
                st.dataframe(pd.DataFrame(score_data), use_container_width=True)

with tab_explorer:
    st.markdown("#### 🧩 Parent-Child Chunking Architecture Explorer")
    st.markdown("""
    Explore how each meeting row is decomposed into **Speaker-Aware Parent Chunks** (~1000 chars) and **Retrieval Child Chunks** (~250 chars).
    """)
    
    if retriever.is_loaded:
        meeting_ids = sorted(list(set([p["meeting_id"] for p in retriever.parents_store.values()])))
        selected_m_id = st.selectbox("Select Meeting ID to Inspect:", options=meeting_ids, index=0)
        
        m_parents = [p for p in retriever.parents_store.values() if p["meeting_id"] == selected_m_id]
        
        if m_parents:
            st.info(f"**Meeting {selected_m_id}** contains **{len(m_parents)} Parent Chunks** across {len(set([p['speaker'] for p in m_parents]))} speakers.")
            
            # Show summary from metadata
            st.markdown(f"**Meeting Summary (Stored in Parent Metadata):**\n> {m_parents[0]['metadata'].get('summary', 'N/A')}")
            
            st.markdown(f"##### Parent Chunks for Meeting {selected_m_id}:")
            for p in m_parents[:10]:
                with st.expander(f"📌 {p['parent_id']} | {p['speaker']} ({len(p['text'])} chars | {len(p['child_ids'])} children)"):
                    st.markdown(f"**Parent Dialogue Context:**\n```\n{p['text']}\n```")
                    st.markdown(f"**Child Chunk IDs linked to this Parent:** `{', '.join(p['child_ids'])}`")
            if len(m_parents) > 10:
                st.caption(f"*Showing first 10 of {len(m_parents)} parent chunks.*")

with tab_dataset:
    st.markdown("#### 📁 Combined Meeting Dataset Statistics")
    
    combined_csv_path = os.path.join(os.path.dirname(__file__), "Data", "combined_meetings.csv")
    if os.path.exists(combined_csv_path):
        df_combined = pd.read_csv(combined_csv_path)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Meetings", len(df_combined))
        col2.metric("Train Split", len(df_combined[df_combined['split'] == 'train']))
        col3.metric("Validation Split", len(df_combined[df_combined['split'] == 'validation']))
        col4.metric("Test Split", len(df_combined[df_combined['split'] == 'test']))
        
        st.markdown("##### Sample Combined Rows:")
        st.dataframe(
            df_combined[["unified_id", "id", "split", "summary"]].head(10),
            use_container_width=True
        )
