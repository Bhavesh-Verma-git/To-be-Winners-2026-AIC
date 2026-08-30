import os
import time
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Suppress tokenizers fork warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

SYSTEM_PROMPT = """You are an expert Customer Decision Support Assistant analyzing corporate product design meetings and discussions.

Your goal is to provide a DIRECT, PRECISE, and FACTUAL answer based ONLY on the provided meeting dialogue chunks.

Instructions:
1. Provide the direct answer in the very first sentence.
2. Be concise, objective, and highlight exact decisions, opinions, or numbers mentioned by speakers (e.g., target costs, demographic age brackets, component choices like LCD vs LED, battery vs solar).
3. If different team members had contrasting views (e.g., Industrial Designer vs Marketing), summarize their positions clearly.
4. Do NOT speculate or invent information not supported by the context.
5. If the context does not contain the answer, state clearly: "Based on the retrieved meeting discussions, this specific topic was not found."
"""

class DecisionSupportAgent:
    def __init__(self, model_name: Optional[str] = None, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model_name = model_name or os.getenv("GROQ_MODEL", "qwen/qwen3.6-27b")
        self._llm = None
        
    def _get_llm(self, model_name: Optional[str] = None, api_key: Optional[str] = None):
        key = api_key or self.api_key or os.getenv("GROQ_API_KEY")
        model = model_name or self.model_name
        
        if not key or key.strip() == "" or key == "your_groq_api_key_here":
            return None
            
        return ChatGroq(
            groq_api_key=key,
            model_name=model,
            temperature=0.1,
            max_tokens=1024,
            timeout=30,
            max_retries=2
        )
        
    def answer_query(
        self,
        query: str,
        retriever,
        top_k: int = 5,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes the full RAG pipeline:
        1. Hybrid retrieval (FAISS + BM25) resolving to parent chunks
        2. Direct answer generation via Groq LLM
        3. Timing and metadata collection
        """
        t_start = time.time()
        
        # 1. Retrieval
        retrieval_output = retriever.search(
            query=query,
            top_k=top_k,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight
        )
        retrieval_time_ms = retrieval_output["latency_ms"]
        context_text = retrieval_output["context_text"]
        results = retrieval_output["results"]
        
        # 2. LLM Generation
        llm = self._get_llm(model_name=model_name, api_key=api_key)
        
        if llm is None:
            answer = (
                "⚠️ **Groq API Key Required**: Please provide your Groq API key in the `.env` file "
                "or enter it in the Streamlit sidebar to generate LLM answers. "
                "\n\n*The hybrid retrieval succeeded and the top matching context chunks are displayed below!*"
            )
            llm_time_ms = 0.0
        else:
            t_llm_start = time.time()
            user_prompt = f"""Context from meeting dialogues:
---
{context_text}
---

Question: {query}

Provide a direct, concise, and factual answer based on the meeting context above:"""
            
            try:
                response = llm.invoke([
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=user_prompt)
                ])
                raw_answer = response.content
                
                # Robust extraction for reasoning/thinking models (e.g. <think>...</think>)
                import re
                if "<think>" in raw_answer:
                    if "</think>" in raw_answer:
                        clean_answer = raw_answer.split("</think>", 1)[1].strip()
                    else:
                        clean_answer = re.sub(r"<think>[\s\S]*?(?:</think>|$)", "", raw_answer).strip()
                    
                    # If clean_answer is empty due to unclosed think block, extract final drafted answer
                    if not clean_answer:
                        matches = re.findall(r'"([^"]{25,})"', raw_answer)
                        if matches:
                            clean_answer = matches[-1].strip()
                        else:
                            clean_answer = raw_answer.replace("<think>", "").strip()
                else:
                    clean_answer = raw_answer.strip()
                    
                answer = clean_answer
                llm_time_ms = round((time.time() - t_llm_start) * 1000, 2)
            except Exception as e:
                answer = f"❌ Error invoking Groq model ({model_name or self.model_name}): {str(e)}"
                llm_time_ms = round((time.time() - t_llm_start) * 1000, 2)
                
        total_time_ms = round((time.time() - t_start) * 1000, 2)
        
        return {
            "query": query,
            "answer": answer,
            "results": results,
            "retrieval_latency_ms": retrieval_time_ms,
            "llm_latency_ms": llm_time_ms,
            "total_latency_ms": total_time_ms,
            "model_used": model_name or self.model_name
        }
