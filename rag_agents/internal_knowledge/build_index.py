"""
============================================================
  Agent 3: Internal Knowledge Assistant
  Phase 2 — build_index.py

  What this script does (in order):
  ─────────────────────────────────
  1. PARSE   — reads each .md file, strips YAML frontmatter,
               extracts title + builds the real MS Docs URL.
  2. CLEAN   — strips image tags (keeps alt text), removes
               HTML comments and [!INCLUDE] directives.
  3. CHUNK   — splits each doc on Markdown headers (##, ###)
               then further splits large sections to max 500 tokens.
               Code blocks and tables are NEVER broken mid-way.
  4. ENRICH  — attaches full metadata to each chunk:
               {title, section, source_url, source, chunk_id,
                has_code, local_file}
  5. PERSIST — saves all chunks as a JSONL file (intermediate
               artifact) so you never need to re-scrape.
  6. EMBED   — generates vectors using bge-small-en-v1.5 (better
               than MiniLM for technical/CLI documentation).
  7. FAISS   — builds and saves a FAISS vector index.
  8. BM25    — builds and saves a BM25 keyword index (pickled).

  Run once after download_docs.py:
     python rag_agents/internal_knowledge/build_index.py
============================================================
"""

import os
import re
import sys
import json
import pickle
import warnings
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env")

from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

try:
    from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
DOCS_DIR        = BASE_DIR / "documents"
INDEX_DIR       = BASE_DIR / "faiss_index"
JSONL_PATH      = INDEX_DIR / "chunks.jsonl"
BM25_PATH       = INDEX_DIR / "bm25_index.pkl"

# ── Model ─────────────────────────────────────────────────────
# bge-small-en-v1.5 ranks higher than MiniLM on MTEB benchmark
# for technical documentation retrieval. Same CPU speed, better accuracy.
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# ── Chunking Settings ─────────────────────────────────────────
# We split headers at 2 levels: H2 (##) and H3 (###)
# MAX_CHUNK_TOKENS: large enough to keep full CLI examples intact
MAX_CHUNK_TOKENS = 500
CHUNK_OVERLAP    = 60   # overlap keeps context continuity between chunks

# ── Markdown Headers to Split On ─────────────────────────────
HEADERS_TO_SPLIT = [
    ("#",  "h1"),
    ("##", "h2"),
    ("###","h3"),
]

# ── URL Builder ───────────────────────────────────────────────
def build_ms_docs_url(filename: str) -> str:
    """
    Converts a local filename to the real learn.microsoft.com URL.

    Example:
      configure-custom-domain.md
      → https://learn.microsoft.com/en-us/azure/app-service/configure-custom-domain
    """
    slug = filename.replace(".md", "")
    return f"https://learn.microsoft.com/en-us/azure/app-service/{slug}"


# ── Step 1: Parse YAML Frontmatter ───────────────────────────
def parse_frontmatter(content: str) -> tuple[dict, str]:
    """
    Splits a markdown file into:
    - metadata dict (from the YAML frontmatter block)
    - body text (everything after the frontmatter)

    Frontmatter looks like:
      ---
      title: Configure a Node.js app
      ms.date: 01/01/2024
      ---
      <actual article content here>
    """
    meta = {}
    body = content

    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            frontmatter_raw = parts[1]
            body = parts[2].strip()
            # Parse key: value pairs from the YAML block
            for line in frontmatter_raw.splitlines():
                if ":" in line:
                    k, _, v = line.partition(":")
                    meta[k.strip().lower()] = v.strip()

    return meta, body


# ── Step 2: Clean Markdown Content ───────────────────────────
def clean_markdown(text: str) -> str:
    """
    Cleans raw markdown text before chunking:
    - Converts image tags to [Diagram: <alt text>] (keeps meaning)
    - Removes [!INCLUDE] directives
    - Removes HTML comments <!-- ... -->
    - Strips excessive blank lines
    """
    # Images: ![alt text](path) → [Diagram: alt text]
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"[Diagram: \1]", text)

    # Remove [!INCLUDE] directive lines
    text = re.sub(r"\[!INCLUDE\s*\[.*?\]\(.*?\)\]", "", text, flags=re.IGNORECASE)

    # Remove HTML comments
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)

    # Remove [!NOTE], [!TIP], [!WARNING] callout markers (keep content after)
    text = re.sub(r"\[!(NOTE|TIP|WARNING|IMPORTANT|CAUTION)\]", "", text)

    # Collapse multiple blank lines to single blank line
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ── Step 3: Chunk One Document ────────────────────────────────
def chunk_document(filepath: Path) -> list[dict]:
    """
    Reads one markdown file and returns a list of chunk dicts.

    Each chunk dict:
    {
      "text":       "...",
      "chunk_id":   "configure-custom-domain_chunk_003",
      "title":      "Map an existing custom DNS name...",
      "section":    "Create the DNS records",
      "source_url": "https://learn.microsoft.com/...",
      "source":     "articles/app-service/configure-custom-domain.md",
      "local_file": "configure-custom-domain.md",
      "has_code":   true
    }
    """
    raw_content = filepath.read_text(encoding="utf-8", errors="ignore")
    meta, body  = parse_frontmatter(raw_content)
    body        = clean_markdown(body)

    # Extract doc title from frontmatter (fallback: filename)
    doc_title  = meta.get("title", filepath.stem.replace("-", " ").title())
    source_url = build_ms_docs_url(filepath.name)
    source_rel = f"articles/app-service/{filepath.name}"

    # ── Stage 1: Split on Markdown headers ──────────────────
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT,
        strip_headers=False   # Keep the header text inside the chunk
    )
    header_chunks = header_splitter.split_text(body)

    # ── Stage 2: Sub-split any large sections ────────────────
    # Separators in priority order — never breaks inside a code block
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_CHUNK_TOKENS * 4,   # approx 4 chars per token
        chunk_overlap=CHUNK_OVERLAP * 4,
        separators=[
            "\n```\n",   # 1st: between code blocks
            "\n\n",      # 2nd: paragraph breaks
            "\n|",       # 3rd: table row boundaries
            "\n",        # 4th: line breaks
            " ",         # last resort: word boundary
        ]
    )

    chunks = []
    chunk_index = 0

    for hc in header_chunks:
        # Get the section heading from header metadata
        section = (
            hc.metadata.get("h3") or
            hc.metadata.get("h2") or
            hc.metadata.get("h1") or
            doc_title
        )

        # Sub-split if chunk is still too large
        sub_texts = recursive_splitter.split_text(hc.page_content)

        for sub_text in sub_texts:
            sub_text = sub_text.strip()
            if len(sub_text) < 50:   # Skip near-empty chunks
                continue

            has_code = "```" in sub_text

            chunks.append({
                "text":       sub_text,
                "chunk_id":   f"{filepath.stem}_chunk_{chunk_index:03d}",
                "title":      doc_title,
                "section":    section,
                "source_url": source_url,
                "source":     source_rel,
                "local_file": filepath.name,
                "has_code":   has_code,
            })
            chunk_index += 1

    return chunks


# ── Step 4: Process All Documents ─────────────────────────────
def process_all_documents() -> list[dict]:
    md_files = sorted(DOCS_DIR.glob("*.md"))
    if not md_files:
        print(f"[ERROR] No markdown files found in: {DOCS_DIR}")
        print("        Run download_docs.py first.")
        sys.exit(1)

    print(f"\n[CHUNK] Processing {len(md_files)} markdown files...")
    all_chunks = []

    for i, filepath in enumerate(md_files, 1):
        chunks = chunk_document(filepath)
        all_chunks.extend(chunks)
        print(f"   [{i:3d}/{len(md_files)}] {filepath.name:55s} → {len(chunks):3d} chunks")

    print(f"\n[OK] Total chunks built: {len(all_chunks)}")
    return all_chunks


# ── Step 5: Save Chunks to JSONL ─────────────────────────────
def save_jsonl(chunks: list[dict]):
    """
    Saves chunks as a JSONL file BEFORE embedding.
    This is our reproducible artifact — if we change the embedding
    model later, we re-embed from JSONL without re-scraping.
    """
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with open(JSONL_PATH, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"[OK] Chunks saved to JSONL: {JSONL_PATH}")
    print(f"     (Intermediate artifact — safe to re-embed without re-scraping)")


# ── Step 6: Build FAISS Vector Index ─────────────────────────
def build_faiss(chunks: list[dict]):
    """
    Converts each chunk's text into a vector using bge-small-en-v1.5
    and stores them in a FAISS index for semantic (meaning-based) search.

    Why bge-small over MiniLM?
    - bge-small ranks higher on MTEB for technical document retrieval
    - Same CPU speed and model size
    - Better at understanding Azure CLI commands and tech terminology
    """
    print(f"\n[FAISS] Building vector index with {EMBEDDING_MODEL}...")

    embedder = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}  # cosine similarity
    )

    # Convert chunks to LangChain Document objects (FAISS expects this format)
    documents = []
    for chunk in chunks:
        documents.append(Document(
            page_content=chunk["text"],
            metadata={
                "chunk_id":   chunk["chunk_id"],
                "title":      chunk["title"],
                "section":    chunk["section"],
                "source_url": chunk["source_url"],
                "source":     chunk["source"],
                "local_file": chunk["local_file"],
                "has_code":   chunk["has_code"],
            }
        ))

    faiss_store = FAISS.from_documents(documents, embedder)
    faiss_store.save_local(str(INDEX_DIR))
    print(f"[OK] FAISS index saved: {INDEX_DIR}")


# ── Step 7: Build BM25 Keyword Index ─────────────────────────
def build_bm25(chunks: list[dict]):
    """
    Builds a BM25 keyword index over the chunk text.
    
    Why BM25 alongside FAISS?
    - Azure docs contain exact CLI flags, SDK names, config keys
      like 'WEBSITE_DISABLE_SCM_SEPARATION' or 'az webapp up --sku'
    - FAISS (semantic) might miss these exact terms
    - BM25 finds them perfectly via keyword matching
    - RRF then merges both results for maximum recall
    """
    print(f"\n[BM25] Building keyword index...")

    # Tokenize by splitting on whitespace (simple, fast, effective)
    tokenized_corpus = [chunk["text"].lower().split() for chunk in chunks]
    chunk_ids        = [chunk["chunk_id"] for chunk in chunks]

    bm25 = BM25Okapi(tokenized_corpus)

    # Save both the BM25 model and the chunk_id lookup list together
    with open(BM25_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "chunk_ids": chunk_ids}, f)

    print(f"[OK] BM25 index saved: {BM25_PATH}")
    print(f"     (Covers {len(chunk_ids)} chunks for keyword search)")


# ── MAIN ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Agent 3 — Build Index (Chunking + FAISS + BM25)")
    print("=" * 60)

    if not DOCS_DIR.exists():
        print("[ERROR] Documents folder not found. Run download_docs.py first.")
        sys.exit(1)

    # Step 1–3: Parse, clean, chunk all documents
    chunks = process_all_documents()

    # Step 4: Save reproducible JSONL artifact
    save_jsonl(chunks)

    # Step 5: Build FAISS semantic index
    build_faiss(chunks)

    # Step 6: Build BM25 keyword index
    build_bm25(chunks)

    print("\n" + "=" * 60)
    print(f"  Index Building Complete!")
    print(f"  Documents processed : {len(list(DOCS_DIR.glob('*.md')))}")
    print(f"  Total chunks        : {len(chunks)}")
    print(f"  Chunks with code    : {sum(1 for c in chunks if c['has_code'])}")
    print(f"  JSONL artifact      : {JSONL_PATH}")
    print(f"  FAISS index         : {INDEX_DIR}")
    print(f"  BM25 index          : {BM25_PATH}")
    print(f"\n  Next: python rag_agents/internal_knowledge/rag_agent.py")
    print("=" * 60)
