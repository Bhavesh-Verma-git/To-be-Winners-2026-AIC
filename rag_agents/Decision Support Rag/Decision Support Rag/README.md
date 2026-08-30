# 🎯 Customer Decision Support RAG Tool

A high-performance, low-latency Decision Support RAG system designed for enterprise meeting transcripts and design discussions (AMI Meeting Corpus).

---

## ⚡ Key Features

1. **Speaker-Aware Parent-Child Chunking**:
   - **Parent Chunks (~1000 chars, 150 overlap)**: Splits dialogue turns cleanly by speaker, preserving conversational meaning and full sentences.
   - **Child Chunks (~250 chars, 40 overlap)**: Fine-grained search units for maximum keyword and semantic match precision.
   - **Metadata Structure**: Each parent chunk holds pure dialogue text as LLM context, with the full meeting executive summary attached in metadata (`summary`).
   - **Dataset Stats**: 279 unified meetings $\rightarrow$ 8,764 Parent Chunks ($\sim 31.4$ per meeting) $\rightarrow$ 44,328 Child Chunks ($\sim 158.9$ per meeting).

2. **Low-Latency Hybrid Retrieval (0.5 BM25 + 0.5 Dense FAISS)**:
   - **Dense Vector Search**: `all-MiniLM-L6-v2` embeddings indexed with FAISS `IndexFlatIP` (L2 normalized cosine similarity).
   - **Sparse Keyword Search**: `BM25Okapi` with min-max normalized scoring.
   - **Hybrid Fusion**: $\text{Score} = 0.5 \times \text{FAISS} + 0.5 \times \text{BM25}$.
   - **Parent Resolution**: Retrieved top-k child matches dynamically resolve to their unique parent dialogue context.
   - **Ultra-Fast Speed**: $\sim 70-90 \text{ ms}$ retrieval latency.

3. **Direct Decision Agent (ChatGroq)**:
   - Powered by `qwen-2.5-32b` (or `qwen-qwq-32b`, `llama-3.3-70b-versatile`, `deepseek-r1-distill-llama-70b`, etc.).
   - Provides direct, factual answers strictly grounded in retrieved meeting dialogue.

4. **Modern Streamlit Dashboard**:
   - 18 Curated Domain Questions with 1-click test chips.
   - Real-time Q&A with latency breakdown badges (Retrieval ms, LLM ms, Total ms).
   - Detailed Context Inspector: View dialogue chunk data, matched child fragments, and meeting summaries.
   - Parent-Child Architecture Explorer: Inspect any meeting ID and see its chunk breakdown.

---

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Set your Groq API Key in .env
cp .env.example .env
# Edit .env and enter your key: GROQ_API_KEY=your_key_here
```

### 2. Run the Streamlit Application
```bash
streamlit run app.py
```

---

## 💡 18 Curated Questions on Dataset

1. **LCD Display**: *What was discussed in the meeting regarding including an LCD screen on the remote control?*
2. **Target Demographic**: *What target demographic age group was decided for the remote control and why?*
3. **Power Source**: *What were the arguments for and against using solar power versus standard batteries?*
4. **Casing & Materials**: *What materials (plastic vs. rubber/titanium) were proposed for the remote's body and grip?*
5. **Shape & Ergonomics**: *What shape was chosen for the remote control (curved/banana shape vs. flat/rectangular)?*
6. **Button Layout**: *What was decided about the number and placement of buttons on the remote?*
7. **Scroll Wheel / Joystick**: *What were the opinions on replacing arrow keys with a scroll wheel or joystick?*
8. **Speech Recognition**: *How did the team evaluate adding speech recognition or voice control features?*
9. **Remote Locator**: *What feature was suggested to help users find a lost remote control in the room?*
10. **Budget & Cost Limits**: *What is the maximum target production cost and selling price for the remote control?*
11. **Corporate Branding**: *What colors and corporate branding requirements were specified by the company?*
12. **Menu & Teletext**: *How did the team decide to handle advanced TV functions like Menu and Teletext?*
13. **Docking Station**: *Was a docking station or recharging cradle considered for the remote control?*
14. **Usability Lab Survey**: *What did the usability survey of 100 participants reveal about existing remote controls?*
15. **Lighting & Illumination**: *Did the team discuss illuminated buttons or glow-in-the-dark features for dark rooms?*
16. **Water Resistance**: *What was discussed regarding making the remote waterproof or splash-resistant?*
17. **Project Manager Role**: *How did the Project Manager allocate design responsibilities among the team members?*
18. **Component Trade-offs**: *What trade-offs were made between incorporating advanced features and staying within the cost budget?*

---

## 📂 Project Structure

```
├── .env.example              # Template for API keys
├── .gitignore                # Ignored sensitive keys & cache
├── requirements.txt          # Python dependencies
├── combine_and_chunk.py      # Dataset merger, Parent-Child chunking, FAISS & BM25 builder
├── retriever.py              # Low-latency HybridRetriever (0.5 BM25 + 0.5 FAISS)
├── agent.py                  # DecisionSupportAgent with ChatGroq & direct answering prompt
├── app.py                    # Streamlit web application
├── test_retrieval.py         # Automated pipeline & latency test script
└── Data/
    ├── train-2.csv           # Raw train split
    ├── validation.csv        # Raw validation split
    ├── test-2.csv            # Raw test split
    ├── combined_meetings.csv # Merged dataset (279 meetings)
    ├── combined_meetings.json# JSON format
    ├── parents_store.json    # 8,764 Parent chunks with summary in metadata
    ├── children_store.json   # 44,328 Child chunks
    ├── faiss_index.bin       # Precomputed FAISS index
    └── bm25_index.pkl        # Serialized BM25 index
```
