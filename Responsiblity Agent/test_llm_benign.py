from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
import os
from dotenv import load_dotenv
load_dotenv()

from src.agent.prompts import RESPONSIBILITY_AGENT_SYSTEM_PROMPT
from src.config import settings
from src.agent.graph import _extract_verdict_from_llm_response

llm = ChatGoogleGenerativeAI(
    model=settings.GEMINI_MODEL,
    temperature=0.0,
    google_api_key=os.getenv('GOOGLE_API_KEY')
)

context_str = """
--- [EVIDENCE CHUNK #1] ---
Heading: Article 5
Text: Prohibits AI systems that deploy subliminal techniques to materially distort a person's behaviour.
"""

query = "all indians are good"

user_message = f"""### STATEMENT TO EVALUATE:
\"\"\"
{query}
\"\"\"

### RETRIEVED EVIDENCE CONTEXT (from EU AI Act, NIST AI RMF, UN/UNESCO Hate Speech frameworks, EU DSA, EEOC, CoE):
{context_str}

IMPORTANT INSTRUCTIONS:
1. First read the STATEMENT carefully.
2. Determine if it contains hate speech, bigotry, discrimination, stereotyping, or harmful AI practices.
3. Use the retrieved context chunks as reference material — they do NOT inherently mean the statement is a violation. You must evaluate the statement on its own merits to determine if it is genuinely harmful or benign.
4. Output your analysis following the strict format specified in your system prompt."""

messages = [
    SystemMessage(content=RESPONSIBILITY_AGENT_SYSTEM_PROMPT),
    HumanMessage(content=user_message)
]

print("Invoking LLM...")
response = llm.invoke(messages)
raw = response.content if hasattr(response, "content") else str(response)
if isinstance(raw, list):
    text = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in raw).strip()
else:
    text = str(raw).strip()

print("\n--- LLM RESPONSE ---")
print(text[:400])
print("\n--- EXTRACTED VERDICT ---")
print("Is Violation?", _extract_verdict_from_llm_response(text))
