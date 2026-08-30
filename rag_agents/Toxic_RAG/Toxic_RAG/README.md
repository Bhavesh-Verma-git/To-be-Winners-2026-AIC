# 🛡️ Toxic RAG: AI Safety Hybrid Retrieval Pipeline

An AI Safety Hybrid Retrieval-Augmented Generation (RAG) system built with **LangChain**, **FAISS (Dense Vector DB)**, **BM25 (Sparse Lexical Search)**, **Groq LLM**, and a **Streamlit Web UI**.

---

## 🚀 Key Features

1. **Cleaned AI Safety Dataset (`Dataset/final_tox_Rag.csv`)**:
   - Merged and cleaned from `annotated_train.csv` and `annotated_test.csv` (9,900 rows).
   - Removed `b''` byte string artifacts, unescaped quote characters, and normalized whitespace.
   - Preserves exact required columns: `Text`, `target group`, `factual`, `in-group effect`, `framing`, `lewd`, `predicted group`, `stereotyping`.

2. **Row-Level Document Indexing**:
   - Each row's `Text` is the full chunk (no arbitrary sub-chunking).
   - All 7 other columns are preserved as rich Document metadata.

3. **50/50 Hybrid Ensemble Retrieval**:
   - **Dense Retrieval**: FAISS with `sentence-transformers/all-MiniLM-L6-v2` embeddings (Weight: 0.5).
   - **Sparse Retrieval**: BM25 Lexical Keyword Search (Weight: 0.5).
   - **Fusion**: Weighted Reciprocal Rank Fusion (RRF) returning the top **7 chunks**.
   - **Persistent Caching**: Pre-computed indices cached in `.index_cache/` for sub-millisecond retrieval.

4. **Groq LLM & Context-Grounded Answer Generation**:
   - Powered by `ChatGroq` (`openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `qwen/qwen3.6-27b`).
   - Answers queries strictly based on the retrieved context and safety metadata.

5. **Streamlit Web Interface (`app.py`)**:
   - Clean, interactive web UI to query the RAG system, view answers, monitor latencies, and inspect all 7 retrieved chunks with complete metadata cards.

---

## 📁 Directory Structure

```
Toxic_RAG/
├── Dataset/
│   ├── annotated_train.csv       # Original train dataset
│   ├── annotated_test.csv        # Original test dataset
│   └── final_tox_Rag.csv         # Cleaned final dataset (9,900 rows)
├── .index_cache/                 # Pre-computed FAISS & BM25 cached indices
│   ├── faiss_index/
│   └── bm25_retriever.pkl
├── app.py                        # Streamlit Web Application
├── prepare_dataset.py            # Dataset cleaning & preparation script
├── rag_agent.py                  # Core Hybrid RAG Agent & LangChain pipeline
├── main.py                       # CLI Interface & benchmark tool
├── requirements.txt              # Dependencies
├── .env                          # Environment configuration (GROQ_API_KEY)
└── README.md
```

---

## 🛠️ Setup & Usage

### 1. Configure Groq API Key
Add your Groq API key to `.env`:
```env
GROQ_API_KEY=gsk_your_groq_api_key_here
```

### 2. Launch Streamlit Web UI
```bash
streamlit run app.py
```

### 3. CLI Interface
**Interactive Mode:**
```bash
python main.py
```

**Single Query Evaluation:**
```bash
python main.py --query "asians taking over the world"
```

### 4. Python API Usage

```python
from rag_agent import ToxicRAGAgent

agent = ToxicRAGAgent()

# Query
result = agent.query("asians taking over the world")

print("Answer:", result["answer"])
print(f"Retrieval Latency: {result['retrieval_latency_ms']} ms")
print(f"Generation Latency: {result['generation_latency_ms']} ms")

# Inspect top 7 chunks
for i, doc in enumerate(result["retrieved_chunks"], 1):
    print(f"[{i}] {doc.page_content} | Metadata: {doc.metadata}")
```
