"""
RAGAS-Style Faithfulness Agent - ControlPlane.ai
Uses Groq LLM directly - no ragas library needed.
Implements claim-decomposition methodology.
"""
from __future__ import annotations
import asyncio, json, logging, os, re, time
from typing import Any
from langchain_groq import ChatGroq

logger = logging.getLogger(__name__)

FAITHFULNESS_PASS = 0.70
FAITHFULNESS_WARN = 0.50
RELEVANCY_PASS    = 0.50


class RAGASAgent:
    """
    RAGAS-style faithfulness agent using direct Groq LLM calls.
    Implements the same methodology as the RAGAS library without
    the library dependency (avoids version conflicts).
    """

    def __init__(self):
        self._llm = None

    def _initialize(self):
        if self._llm: return
        key = os.getenv("GROQ_API_KEY")
        if not key: raise EnvironmentError("GROQ_API_KEY not set")
        self._llm = ChatGroq(
            model_name="qwen/qwen3.6-27b",
            temperature=0.0, max_tokens=2048, api_key=key
        )
        logger.info("[RAGASAgent] qwen/qwen3.6-27b loaded via Groq.")

    def _call(self, prompt):
        try:
            raw = self._llm.invoke(prompt).content.strip()
            # Strip <think> tags if present (Qwen is a reasoning model)
            if "</think>" in raw:
                raw = raw.split("</think>")[-1].strip()
            return raw
        except Exception as e:
            logger.warning(f"[RAGASAgent] LLM call failed: {e}")
            return ""

    def _extract_claims(self, answer):
        prompt = (
            "Extract all individual factual claims from the answer below.\n"
            "Return ONLY a valid JSON array of strings, one claim per item.\n\n"
            f"Answer: {answer}\n\nJSON array:"
        )
        raw = self._call(prompt)
        try:
            m = re.search(r"\[.*?\]", raw, re.DOTALL)
            if m: return json.loads(m.group())
        except Exception:
            pass
        # Fallback: sentence split
        return [s.strip() for s in answer.split(".") if len(s.strip()) > 10][:8]

    def _verify_claim(self, context, claim):
        prompt = (
            f"Context: {context[:3000]}\n\n"
            f"Claim: {claim}\n\n"
            "Is this claim fully supported by the context above?\n"
            "Answer with exactly one word: YES or NO."
        )
        return self._call(prompt).upper().startswith("YES")

    def _score_relevancy(self, question, answer):
        prompt = (
            f"Question: {question}\n"
            f"Answer: {answer}\n\n"
            "Rate how well the answer addresses the question.\n"
            "Return ONLY a decimal number from 0.0 to 1.0."
        )
        raw = self._call(prompt)
        try:
            m = re.search(r"\d+\.?\d*", raw)
            return max(0.0, min(1.0, float(m.group()))) if m else 0.5
        except Exception:
            return 0.5

    def score_sync(self, query, rag_answer, retrieved_context):
        """Main synchronous scorer with RAGAS claim decomposition."""
        self._initialize()
        t0 = time.perf_counter()
        ctx = " ".join(retrieved_context) if isinstance(retrieved_context, list) else retrieved_context

        claims = self._extract_claims(rag_answer)
        logger.info(f"[RAGASAgent] Extracted {len(claims)} claims.")

        if not claims:
            faith = 0.5
        else:
            supported = sum(1 for c in claims if self._verify_claim(ctx, c))
            faith = supported / len(claims)
            logger.info(f"[RAGASAgent] {supported}/{len(claims)} claims supported.")

        rel = self._score_relevancy(query, rag_answer)
        ms = (time.perf_counter() - t0) * 1000
        verdict, reason = self._verdict(faith, rel)
        logger.info(f"[RAGASAgent] faith={faith:.3f} rel={rel:.3f} verdict={verdict} ({ms:.0f}ms)")

        return {
            "ragas_scores": {
                "faithfulness": round(faith, 4),
                "answer_relevancy": round(rel, 4)
            },
            "ragas_verdict": verdict,
            "ragas_reasoning": reason,
            "ragas_latency_ms": round(ms, 1)
        }

    async def score_async(self, query, rag_answer, retrieved_context):
        """Async wrapper - runs sync scorer in executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.score_sync, query, rag_answer, retrieved_context
        )

    @staticmethod
    def _verdict(f, r):
        if f >= 0.70:
            if r >= 0.50:
                return "pass", f"Faithfulness={f:.2f} + Relevancy={r:.2f}: Answer is grounded and relevant."
            return "uncertain", f"Faithfulness={f:.2f} OK but Relevancy={r:.2f} is low."
        if f >= 0.50:
            return "uncertain", f"Faithfulness={f:.2f}: Partial hallucination suspected."
        return "fail", f"Faithfulness={f:.2f}: Answer contradicts retrieved context."


# LangGraph node
_agent = RAGASAgent()


async def ragas_node(state: dict) -> dict:
    """
    Async LangGraph node for the Performance Branch.
    Reads:  state[updated_query], state[rag_answer], state[retrieved_context]
    Writes: state[ragas_scores], state[ragas_verdict], state[ragas_reasoning]
    """
    result = await _agent.score_async(
        query=state.get("updated_query") or state.get("user_query", ""),
        rag_answer=state.get("rag_answer", ""),
        retrieved_context=state.get("retrieved_context", []),
    )
    return {**state, **result}


if __name__ == "__main__":
    from pathlib import Path
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parents[2] / ".env")
    logging.basicConfig(level=logging.INFO)
    agent = RAGASAgent()

    print("=" * 60)
    print("TEST 1: Clean faithful answer (expected: PASS)")
    print("=" * 60)
    r = agent.score_sync(
        query="What is the return policy?",
        rag_answer="Products can be returned within 30 days with a receipt.",
        retrieved_context=["Our return policy: return within 30 days with a valid receipt."]
    )
    print(json.dumps(r, indent=2))

    print("\n" + "=" * 60)
    print("TEST 2: Hallucinated answer (expected: FAIL)")
    print("=" * 60)
    r = agent.score_sync(
        query="What is the return window?",
        rag_answer="You can return products within 90 days and get free shipping.",
        retrieved_context=["Our return policy: return within 30 days with a valid receipt."]
    )
    print(json.dumps(r, indent=2))
