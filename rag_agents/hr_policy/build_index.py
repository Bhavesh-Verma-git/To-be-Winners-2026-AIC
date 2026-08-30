"""
============================================================
  HR Policy RAG Agent â€” Step 1: Build Index
  Chunking Strategy: PARENT-CHILD (Hierarchical)

  HOW IT WORKS:
  - PARENT: Full policy section (600-1000 tokens) â†’ stored in dict
  - CHILD:  Small sub-chunk (150-250 tokens)       â†’ stored in FAISS
  - At retrieval: FAISS finds CHILD â†’ system fetches its PARENT
  - LLM reads the full PARENT for complete context

  Run ONCE to build the index.
============================================================
"""

import os
import re
import json
import sys
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import pickle
import warnings
from pathlib import Path
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

load_dotenv(Path(__file__).parent.parent.parent / ".env")

import pdfplumber
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

# â”€â”€ Paths â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
BASE_DIR        = Path(__file__).parent
PDF_PATH        = BASE_DIR / "HR Policy _ KESPL.pdf"
FAISS_INDEX_DIR = BASE_DIR / "faiss_index"
PARENT_STORE    = BASE_DIR / "faiss_index" / "parent_store.json"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# â”€â”€ Known Section Headers in the PDF â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
SECTION_HEADERS = [
    "HISTORICAL BACKGROUND", "HISTORICALBACKGROUND",
    "MISSION", "VISION", "EMPLOYMENT AT KAMAIAH",
    "EMPLOYMENTATKAMAIAHENGINEERINGSERVICESPRIVATELIMITED",
    "ORIENTATION", "EMPLOYMENT CATEGORIES", "EMPLOYMENTCATEGORIES",
    "PERSONNEL DATA", "PERSONNEL", "PROMOTION", "SALARY",
    "SANCTIONING AUTHORITY", "SANCTIONINGAUTHORITY",
    "CASUAL LEAVE", "SICK LEAVE", "PRIVILEGE LEAVE",
    "DRESS CODE", "PERSONAL COMMUNICATIONS", "PHONE",
    "EMAIL POLICY", "INTERNET", "DISCIPLINE",
    "LATE COMING", "LATECOMING", "ATTENDANCE",
    "TERMINATION", "RESIGNATION", "TRAVEL", "ALLOWANCE",
    "HEALTH", "SAFETY", "SMOKING", "MANUAL HANDLING",
    "HUMAN RESOURCES MANUAL"
]

# ── Step 1: Helper — Convert raw table rows to Markdown ──────
def table_to_markdown(table: list) -> str:
    """
    Takes a raw pdfplumber table (list of lists) and converts it
    to a clean GitHub-flavoured markdown table string.

    Example input:
      [["Grade", "Transport", "Daily Allowance", "Accommodation"],
       ["A (Directors)", "Flight", "Rs.500", "Rs.1500"]]

    Example output:
      | Grade | Transport | Daily Allowance | Accommodation |
      |---|---|---|---|
      | A (Directors) | Flight | Rs.500 | Rs.1500 |
    """
    if not table or not table[0]:
        return ""

    # Clean each cell: remove None, strip whitespace
    cleaned = []
    for row in table:
        cleaned_row = [str(cell).strip() if cell is not None else "" for cell in row]
        cleaned.append(cleaned_row)

    # First row = header
    header = "| " + " | ".join(cleaned[0]) + " |"
    divider = "| " + " | ".join(["---"] * len(cleaned[0])) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in cleaned[1:]]

    return "\n".join([header, divider] + rows)

# ── Step 2: Extract Text + Tables from PDF ───────────────────
def extract_text_from_pdf():
    print("[PDF] Extracting text from HR Policy PDF...")
    pages_text = []

    with pdfplumber.open(PDF_PATH) as pdf:
        total = len(pdf.pages)
        print(f"   Total pages: {total}")
        for i, page in enumerate(pdf.pages):

            # ── A: Extract plain text ────────────────────────
            text = page.extract_text() or ""
            text = text.encode("ascii", errors="ignore").decode("ascii")
            text = re.sub(r"\s+", " ", text).strip()

            # ── B: Detect and extract tables (NEW) ───────────
            tables = page.extract_tables()
            table_markdown_blocks = []
            if tables:
                for table in tables:
                    md = table_to_markdown(table)
                    if md:
                        table_markdown_blocks.append(md)
                        print(f"   [OK] Table found on page {i+1} "
                              f"({len(table)} rows) - formatted as Markdown")

            # ── C: Combine text + markdown tables ────────────
            page_content = text
            if table_markdown_blocks:
                page_content += "\n\n" + "\n\n".join(table_markdown_blocks)

            if len(page_content.strip()) > 50:
                pages_text.append({"page": i + 1, "text": page_content})

    full_text = "\n\n".join([p["text"] for p in pages_text])
    print(f"   [OK] Extracted {len(full_text)} characters from {len(pages_text)} pages")
    return full_text, pages_text

# ── Step 2: Build PARENT Chunks (Section Level) ────────────
def build_parent_chunks(full_text: str):
    """
    PARENT = Full policy section (600-1000 tokens)
    We split on clear section header patterns found in the PDF.
    Each parent has:
      - doc_id: unique identifier
      - content: full section text
      - section_title: detected heading
    """
    print("\nðŸ—‚ï¸  Building PARENT chunks (section level)...")

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # ~750 words per parent section
        chunk_overlap=100,    # Small overlap to avoid edge cuts
        separators=["\n\n", "\n", ". ", " "],
    )

    raw_parents = parent_splitter.split_text(full_text)
    parents = {}

    for i, text in enumerate(raw_parents):
        doc_id = f"parent_{i:04d}"

        # Try to detect section title from the first line
        first_line = text.strip().split("\n")[0][:80]
        section_title = first_line if len(first_line) > 5 else f"Section {i+1}"

        parents[doc_id] = {
            "doc_id": doc_id,
            "content": text,
            "section_title": section_title,
            "source": "documents/HR_Policy_KESPL.pdf"
        }

    print(f"   âœ… Built {len(parents)} parent chunks")
    return parents

# â”€â”€ Step 3: Build CHILD Chunks (Sub-section Level) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def build_child_chunks(parents: dict):
    """
    CHILD = Small precise snippet (150-250 tokens)
    Each child:
      - Stores its parent_id so we can fetch parent at retrieval
      - Has section_title prepended (Contextual Prepending technique)
    """
    print("\nðŸ”— Building CHILD chunks (sub-section level)...")

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=250,       # ~180 words per child (precise for search)
        chunk_overlap=30,     # Small overlap so facts aren't split
        separators=["\n\n", "\n", ". ", " "],
    )

    child_docs = []
    for parent_id, parent in parents.items():
        child_texts = child_splitter.split_text(parent["content"])

        for j, child_text in enumerate(child_texts):
            # Contextual Prepending: add section title so child knows its context
            contextual_content = (
                f"[Section: {parent['section_title']}]\n{child_text}"
            )

            child_doc = Document(
                # CHUNK: Small child text (what FAISS searches)
                page_content=contextual_content,

                # METADATA: Links back to its parent for full context
                metadata={
                    "parent_id":     parent_id,
                    "section_title": parent["section_title"],
                    "source":        parent["source"],
                    "chunk_index":   j
                }
            )
            child_docs.append(child_doc)

    print(f"   âœ… Built {len(child_docs)} child chunks (from {len(parents)} parents)")

    # Show example Parent-Child pair
    example_parent_id = list(parents.keys())[2]
    example_child = [c for c in child_docs if c.metadata["parent_id"] == example_parent_id]
    if example_child:
        print(f"\n   --- EXAMPLE PARENT-CHILD PAIR ---")
        print(f"   PARENT [{example_parent_id}]: {parents[example_parent_id]['content'][:150]}...")
        print(f"   CHILD  [{example_parent_id}_c0]: {example_child[0].page_content[:150]}...")

    return child_docs

# â”€â”€ Step 4: Build FAISS on CHILD Chunks â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def build_faiss(child_docs):
    print(f"\nðŸ”¢ Building FAISS index on {len(child_docs)} child chunks...")
    embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    faiss_store = FAISS.from_documents(child_docs, embedder)
    faiss_store.save_local(str(FAISS_INDEX_DIR))
    print(f"   âœ… FAISS index saved â†’ {FAISS_INDEX_DIR}")
    return faiss_store

# â”€â”€ Step 5: Save PARENT Store to Disk â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def save_parent_store(parents: dict):
    with open(PARENT_STORE, "w", encoding="utf-8") as f:
        json.dump(parents, f, indent=2, ensure_ascii=False)
    print(f"   âœ… Parent store saved â†’ {PARENT_STORE}")
    print(f"      ({len(parents)} parent sections stored)")

# â”€â”€ MAIN â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if __name__ == "__main__":
    print("=" * 62)
    print("  HR Policy RAG â€” Index Builder (Parent-Child Chunking)")
    print("=" * 62)

    # Check if index already exists
    if (FAISS_INDEX_DIR / "index.faiss").exists() and PARENT_STORE.exists():
        print("\nâš¡ Index already exists! Skipping rebuild.")
        print("   Delete faiss_index folder to force rebuild.")
        import sys; sys.exit(0)

    # Create output dir
    FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # Build step by step
    full_text, pages  = extract_text_from_pdf()
    parents           = build_parent_chunks(full_text)
    child_docs        = build_child_chunks(parents)
    build_faiss(child_docs)
    save_parent_store(parents)

    print("\n" + "=" * 62)
    print("  âœ… Index Building Complete!")
    print("  Now run: python rag_agent.py")
    print("=" * 62)

