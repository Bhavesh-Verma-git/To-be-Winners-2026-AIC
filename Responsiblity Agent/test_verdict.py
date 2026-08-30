from src.agent.graph import _extract_verdict_from_llm_response
cases = [
    "### 1. 🏷️ COMPLIANCE VERDICT\n- **STATUS**: COMPLIANT ✅\nThis statement is not UNETHICAL.",
    "### 1. 🏷️ COMPLIANCE VERDICT\n- **STATUS**: UNETHICAL / NON-COMPLIANT ⚠️ (FLAGGED)\nThis is bad."
]
for c in cases:
    print(f"Is Violation? {_extract_verdict_from_llm_response(c)}")
