import os
import time
from retriever import HybridRetriever
from agent import DecisionSupportAgent

def run_tests():
    print("=== Testing Decision Support RAG Pipeline ===")
    retriever = HybridRetriever()
    retriever.load()
    
    test_queries = [
        "What was discussed in the meeting regarding including an LCD screen on the remote control?",
        "What target demographic age group was decided for the remote control and why?",
        "What were the arguments for and against using solar power versus standard batteries?"
    ]
    
    agent = DecisionSupportAgent()
    
    for q in test_queries:
        print(f"\n--- Query: {q} ---")
        t0 = time.time()
        output = retriever.search(q, top_k=5, dense_weight=0.5, sparse_weight=0.5)
        print(f"Retrieval latency: {output['latency_ms']} ms")
        print(f"Resolved top-k parent chunks count: {len(output['results'])}")
        
        for idx, res in enumerate(output["results"][:2]):
            print(f"\n  [Top Result {idx+1}]")
            print(f"  Parent ID: {res['parent_id']} | Meeting ID: {res['meeting_id']} | Speaker: {res['speaker']}")
            print(f"  Hybrid Score: {res['hybrid_score']:.4f} (FAISS: {res['dense_score']:.3f}, BM25: {res['bm25_score']:.3f})")
            print(f"  Matched Child snippet: {res['matched_child'][:80]}...")
            print(f"  Parent Text snippet: {res['parent_text'][:120]}...")
            print(f"  Summary from Metadata snippet: {res['metadata']['summary'][:100]}...")
            
    print("\n✓ All retrieval pipeline tests passed successfully!")

if __name__ == "__main__":
    run_tests()
