#!/usr/bin/env python3
"""
Ingestion Runner Script for the Responsibility Agent
Parses NIST.AI.100-1.pdf and EU AI Act PDF, builds hierarchical chunks,
and indexes them across Vector DB (Chroma), BM25, and Neo4j Knowledge Graph.
"""

import sys
import argparse
from src.pipeline import ResponsibilityPipeline

def main():
    parser = argparse.ArgumentParser(description="Ingest compliance documents into Hybrid Stores.")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild of chunks and indices")
    args = parser.parse_args()

    print("=" * 70)
    print("🚀 RESPONSIBILITY AGENT: INGESTION PIPELINE")
    print("=" * 70)
    
    pipeline = ResponsibilityPipeline()
    pipeline.ingest(force_rebuild=args.rebuild)
    
    print("\n✅ Ingestion complete!")
    print(f"Total Chunks in Store: {len(pipeline.chunk_store)}")
    print("=" * 70)

if __name__ == "__main__":
    main()
