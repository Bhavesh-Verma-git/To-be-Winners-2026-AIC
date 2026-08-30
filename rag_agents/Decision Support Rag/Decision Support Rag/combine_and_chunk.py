import os
import json
import pickle
import time
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = os.path.join(os.path.dirname(__file__), "Data")
PARENT_CHUNK_SIZE = 1000
PARENT_CHUNK_OVERLAP = 150
CHILD_CHUNK_SIZE = 250
CHILD_CHUNK_OVERLAP = 40
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

def load_and_combine_data():
    """Load train, validation, and test datasets and combine them."""
    print("-> Loading dataset files...")
    files = {
        "train": os.path.join(DATA_DIR, "train-2.csv"),
        "validation": os.path.join(DATA_DIR, "validation.csv"),
        "test": os.path.join(DATA_DIR, "test-2.csv")
    }
    
    dfs = []
    for split, path in files.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["split"] = split
            dfs.append(df)
            print(f"   Loaded {split}: {len(df)} rows")
        else:
            print(f"   Warning: File {path} not found.")
            
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df["unified_id"] = combined_df.apply(lambda r: f"M_{r.name:03d}_id{r['id']}", axis=1)
    print(f"-> Combined total rows: {len(combined_df)}")
    
    # Save combined dataset
    combined_csv = os.path.join(DATA_DIR, "combined_meetings.csv")
    combined_json = os.path.join(DATA_DIR, "combined_meetings.json")
    combined_df.to_csv(combined_csv, index=False)
    
    records = combined_df.to_dict(orient="records")
    with open(combined_json, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"-> Saved combined data to {combined_csv} and {combined_json}")
    
    return combined_df

def create_parent_child_chunks(df):
    """
    Performs speaker-aware parent-child chunking.
    - Each row's dialogue is split speaker-by-speaker.
    - Speaker utterances are chunked into Parent Chunks (~1000 chars, 150 overlap).
    - Parent chunks store dialogue text as context, and meeting summary in metadata.
    - Child Chunks (~250 chars, 40 overlap) are created for granular retrieval indexing.
    """
    print("-> Creating Parent-Child chunks...")
    
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=PARENT_CHUNK_OVERLAP,
        separators=[". ", "? ", "! ", "\n", " ", ""]
    )
    
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
        separators=[". ", "? ", "! ", "\n", " ", ""]
    )
    
    parents_store = {}
    children_store = []
    
    row_stats = []
    
    for row_idx, row in df.iterrows():
        meeting_id = row["id"]
        unified_id = row["unified_id"]
        summary = str(row["summary"]) if pd.notna(row["summary"]) else ""
        dialogue = str(row["dialogue"]) if pd.notna(row["dialogue"]) else ""
        split_name = row["split"]
        
        speaker_lines = dialogue.split("\n")
        row_parent_count = 0
        row_child_count = 0
        
        for s_idx, s_line in enumerate(speaker_lines):
            s_line = s_line.strip()
            if not s_line:
                continue
            
            speaker_name = s_line.split(":", 1)[0].strip() if ":" in s_line else f"Speaker_{s_idx}"
            parent_texts = parent_splitter.split_text(s_line)
            
            for p_idx, p_text in enumerate(parent_texts):
                parent_id = f"P_{row_idx}_{s_idx}_{p_idx}"
                
                # Context is strictly chunk text; Summary is metadata
                parent_obj = {
                    "parent_id": parent_id,
                    "meeting_id": int(meeting_id) if str(meeting_id).isdigit() else str(meeting_id),
                    "unified_id": unified_id,
                    "speaker": speaker_name,
                    "text": p_text,
                    "metadata": {
                        "meeting_id": int(meeting_id) if str(meeting_id).isdigit() else str(meeting_id),
                        "unified_id": unified_id,
                        "split": split_name,
                        "speaker": speaker_name,
                        "summary": summary
                    },
                    "child_ids": []
                }
                
                # Split parent into child chunks
                child_texts = child_splitter.split_text(p_text)
                for c_idx, c_text in enumerate(child_texts):
                    child_id = f"C_{parent_id}_{c_idx}"
                    parent_obj["child_ids"].append(child_id)
                    
                    children_store.append({
                        "child_id": child_id,
                        "parent_id": parent_id,
                        "meeting_id": parent_obj["meeting_id"],
                        "unified_id": unified_id,
                        "speaker": speaker_name,
                        "text": c_text
                    })
                    row_child_count += 1
                    
                parents_store[parent_id] = parent_obj
                row_parent_count += 1
                
        row_stats.append({
            "row_idx": row_idx,
            "unified_id": unified_id,
            "meeting_id": meeting_id,
            "parent_chunks": row_parent_count,
            "child_chunks": row_child_count
        })
        
    print(f"-> Total Parent Chunks: {len(parents_store)}")
    print(f"-> Total Child Chunks: {len(children_store)}")
    print(f"-> Avg Parents per Meeting: {len(parents_store) / len(df):.1f}")
    print(f"-> Avg Children per Meeting: {len(children_store) / len(df):.1f}")
    
    # Save parent store and children store
    parents_file = os.path.join(DATA_DIR, "parents_store.json")
    children_file = os.path.join(DATA_DIR, "children_store.json")
    
    with open(parents_file, "w", encoding="utf-8") as f:
        json.dump(parents_store, f, indent=2, ensure_ascii=False)
        
    with open(children_file, "w", encoding="utf-8") as f:
        json.dump(children_store, f, indent=2, ensure_ascii=False)
        
    print(f"-> Saved stores to {parents_file} and {children_file}")
    return parents_store, children_store

def build_faiss_and_bm25_indices(children_store):
    """
    Builds and saves:
    1. FAISS dense vector index using sentence-transformers (Cosine / Inner Product with L2 norm)
    2. BM25 sparse keyword search index
    """
    print(f"-> Loading embedding model '{EMBEDDING_MODEL_NAME}'...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    texts = [c["text"] for c in children_store]
    print(f"-> Generating embeddings for {len(texts)} child chunks...")
    t0 = time.time()
    embeddings = embed_model.encode(texts, batch_size=128, show_progress_bar=True, convert_to_numpy=True)
    
    # L2 normalize embeddings for cosine similarity via inner product
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"-> FAISS index built with {index.ntotal} vectors in {time.time() - t0:.2f}s")
    
    faiss_file = os.path.join(DATA_DIR, "faiss_index.bin")
    faiss.write_index(index, faiss_file)
    print(f"-> Saved FAISS index to {faiss_file}")
    
    # Build BM25 index
    print("-> Tokenizing texts for BM25...")
    tokenized_corpus = [text.lower().split() for text in texts]
    bm25 = BM25Okapi(tokenized_corpus)
    
    bm25_file = os.path.join(DATA_DIR, "bm25_index.pkl")
    with open(bm25_file, "wb") as f:
        pickle.dump({"bm25": bm25, "tokenized_corpus": tokenized_corpus}, f)
    print(f"-> Saved BM25 index to {bm25_file}")
    
    print("All indices successfully built and saved!")

if __name__ == "__main__":
    df = load_and_combine_data()
    parents_store, children_store = create_parent_child_chunks(df)
    build_faiss_and_bm25_indices(children_store)
