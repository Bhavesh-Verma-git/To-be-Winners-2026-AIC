#!/usr/bin/env python3
"""
main.py
CLI Interface for Toxic RAG Agent.
70% FAISS (Dense Vector) + 30% BM25 (Sparse Lexical), Top 7 chunks, Groq LLM.
"""

import sys
import os
import argparse
from rag_agent import ToxicRAGAgent

def print_banner():
    print("=" * 70)
    print("      🛡️  TOXIC RAG AGENT - AI SAFETY HYBRID RETRIEVAL PIPELINE 🛡️")
    print("=" * 70)
    print("  • Architecture: 70% Dense Vector (FAISS) + 30% Sparse Lexical (BM25)")
    print("  • Retrieval Count: Top 7 Full Row Chunks with Complete Safety Metadata")
    print("  • LLM Engine: Groq Ultra-Low Latency Inference")
    print("=" * 70)


def display_results(result: dict, show_metadata: bool = True):
    print("\n" + "-" * 70)
    print(f"📌 QUERY: {result['query']}")
    print("-" * 70)
    print(f"⏱️  LATENCY METRICS:")
    print(f"   • Hybrid Retrieval : {result['retrieval_latency_ms']:.2f} ms")
    print(f"   • Groq Generation  : {result['generation_latency_ms']:.2f} ms")
    print(f"   • Total Pipeline   : {result['total_latency_ms']:.2f} ms")
    print("-" * 70)
    print("🤖 DIRECT DATABASE ANSWER:")
    print(result['answer'])
    print("-" * 70)
    print("📚 RETRIEVED BENCHMARK CHUNKS (Top 7 Weighted Ensemble: 70% FAISS / 30% BM25):")
    for i, doc in enumerate(result['retrieved_chunks'], 1):
        meta = doc.metadata
        print(f"\n[{i}] Text: \"{doc.page_content}\"")
        if show_metadata:
            print(f"    ├─ Target Group   : {meta.get('target group', '')}")
            print(f"    ├─ Factual Status : {meta.get('factual', '')}")
            print(f"    ├─ In-Group Effect: {meta.get('in-group effect', '')}")
            print(f"    ├─ Framing        : {meta.get('framing', '')}")
            print(f"    ├─ Lewdness       : {meta.get('lewd', '')}")
            print(f"    ├─ Predicted Group: {meta.get('predicted group', '')}")
            print(f"    ├─ Stereotyping   : {meta.get('stereotyping', '')}")
            print(f"    └─ Hybrid Score   : {meta.get('retrieval_score', '')} (Sources: {', '.join(meta.get('matched_retrievers', []))})")
    print("=" * 70 + "\n")


def run_interactive(agent: ToxicRAGAgent):
    print_banner()
    print("\nEnter a text or prompt to query the AI safety database.")
    print("Type 'exit' or 'quit' to end.\n")

    while True:
        try:
            query_text = input("\n🛡️ Enter Query > ").strip()
            if not query_text:
                continue
            if query_text.lower() in ("exit", "quit", "q"):
                print("Exiting Toxic RAG Agent. Goodbye!")
                break

            result = agent.query(query_text)
            display_results(result)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\n[Error] {e}")


def main():
    parser = argparse.ArgumentParser(description="Toxic RAG Hybrid AI Safety Agent")
    parser.add_argument("--query", "-q", type=str, help="Single query text to evaluate")
    parser.add_argument("--rebuild-index", action="store_true", help="Force rebuild of FAISS and BM25 indices")
    parser.add_argument("--top-k", type=int, default=7, help="Number of retrieved chunks (default: 7)")
    parser.add_argument("--model", type=str, default=None, help="Groq model override (e.g. openai/gpt-oss-120b, openai/gpt-oss-20b)")
    args = parser.parse_args()

    kwargs = {
        "top_k": args.top_k,
        "vector_weight": 0.7,
        "bm25_weight": 0.3,
        "force_rebuild_index": args.rebuild_index
    }
    if args.model:
        kwargs["groq_model"] = args.model

    agent = ToxicRAGAgent(**kwargs)

    if args.query:
        result = agent.query(args.query)
        display_results(result)
    else:
        run_interactive(agent)


if __name__ == "__main__":
    main()
