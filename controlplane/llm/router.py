"""
LiteLLM routing layer - the ONLY place the package talks to an LLM.

Design
------
* One `litellm.Router` instance (lazy singleton).
* For every model category (`light`, `medium`, `heavy`, `judge`, `main_agent`,
  `suggestion`, `responsibility`) we register one deployment per (base model x
  provider key). LiteLLM then load-balances across keys (`simple-shuffle`) and
  fails over across every deployment of the category (`num_retries`), which gives
  us both multi-key parallelism and cross-provider fallback for free.
* `complete()` / `stream_complete()` return an `LLMResult` carrying the text plus
  the metadata the dashboard/LangSmith need: real model id, category, tier, token
  counts, latency, and cost (Groq forced to 0).

Placeholder mode
----------------
If no keys are configured (or `CP_LLM_MOCK=1`) every call returns a deterministic
mock so the whole graph, the tests and the latency benchmark run end-to-end
without credentials. Real keys switch it off automatically.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

from controlplane.config import settings


class LLMUnavailable(RuntimeError):
    """Raised when a real LLM call is required but no provider is usable."""


@dataclass
class LLMResult:
    text: str
    model: str
    category: str
    tier: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    provider: str = "unknown"
    mocked: bool = False
    raw: Any = field(default=None, repr=False)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def as_call_record(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "model": self.model,
            "tier": self.tier,
            "provider": self.provider,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "latency_ms": round(self.latency_ms, 1),
            "cost_usd": round(self.cost_usd, 6),
            "mocked": self.mocked,
        }


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _provider_of(model_id: str) -> str:
    head = model_id.split("/", 1)[0].lower()
    if head in {"groq", "gemini", "openai", "anthropic", "vertex_ai"}:
        return head
    return head


def model_tier(model_id: str) -> int:
    """Coarse capability tier (1..3). Compatible with the XGBoost feature convention."""
    m = model_id.lower()
    if any(t in m for t in ("gpt-4", "2.5-pro", "-pro", "opus", "compound")):
        return 3
    if any(t in m for t in ("70b", "120b", "gpt-3.5", "2.5-flash", "1.5-pro", "gemma2")):
        return 2
    return 1


_HASH_RE = re.compile(r"^[0-9a-f]{16,}$", re.IGNORECASE)
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def _strip_think(text: str) -> str:
    """Remove <think>...</think> reasoning traces (qwen/compound style).
    If that leaves nothing (model put the whole answer inside <think>), fall back
    to the reasoning text with only the tags removed - better than an empty answer."""
    if not text:
        return text
    raw = text
    if "</think>" in text:
        text = text.split("</think>")[-1]
    text = _THINK_RE.sub("", text)
    text = re.sub(r"^\s*<think>.*", "", text, flags=re.DOTALL).strip()
    if not text and raw.strip():
        text = re.sub(r"</?think>", "", raw).strip()
    return text


def _real_model_name(*candidates) -> str:
    """Return the first candidate that looks like a real model id (not a Router hash)."""
    for c in candidates:
        if c and not _HASH_RE.match(str(c)):
            return str(c)
    return str(next((c for c in candidates if c), "unknown"))


def _is_mock() -> bool:
    if os.getenv("CP_LLM_MOCK", "").strip().lower() in {"1", "true", "yes"}:
        return True
    return not settings.has_any_llm()


# --------------------------------------------------------------------------------------
# Router construction
# --------------------------------------------------------------------------------------
_router = None
_router_built = False


def _build_model_list() -> List[Dict[str, Any]]:
    model_list: List[Dict[str, Any]] = []
    for category, base_models in settings.model_catalog.items():
        for base in base_models:
            provider = _provider_of(base)
            if provider == "groq":
                keys = settings.groq_keys
            elif provider in {"gemini", "vertex_ai"}:
                keys = settings.gemini_keys
            else:
                keys = [None]  # let litellm read env
            for key in keys or []:
                params: Dict[str, Any] = {"model": base}
                if key:
                    params["api_key"] = key
                # NOTE: do NOT pass a custom "model_info" - LiteLLM validates it with a
                # pydantic model (e.g. `tier` must be 'free'|'paid'); our metadata is
                # tracked separately in LLMResult.
                model_list.append({"model_name": category, "litellm_params": params})
    return model_list


def get_router():
    """Lazily build (or return) the singleton litellm.Router. None in mock mode."""
    global _router, _router_built
    if _router_built:
        return _router
    _router_built = True

    if _is_mock():
        _router = None
        return None

    import litellm
    from litellm import Router

    litellm.drop_params = True
    litellm.set_verbose = False
    # We call LiteLLM directly (not via LangChain) so its own async langsmith
    # batch-logger only causes "no running event loop" errors in our sync worker
    # threads. LangSmith tracing of the GRAPH still works via LangGraph's native
    # integration; per-model cost/latency is tracked in LLMResult + state instead.
    for attr in ("success_callback", "failure_callback", "callbacks", "_async_success_callback"):
        cur = getattr(litellm, attr, None)
        if isinstance(cur, list):
            setattr(litellm, attr, [c for c in cur if "langsmith" not in str(c).lower()])

    model_list = _build_model_list()
    if not model_list:
        _router = None
        return None

    _router = Router(
        model_list=model_list,
        routing_strategy="simple-shuffle",
        num_retries=settings.litellm_num_retries,
        timeout=settings.litellm_timeout,
        allowed_fails=3,
        cooldown_time=20,
    )
    return _router


def reset_router() -> None:
    """Test helper - force the next get_router() to rebuild."""
    global _router, _router_built
    _router, _router_built = None, False


# --------------------------------------------------------------------------------------
# Mock responses
# --------------------------------------------------------------------------------------
def _mock_text(category: str, messages: List[Dict[str, str]], response_format: Optional[dict]) -> str:
    user = ""
    system = ""
    for m in reversed(messages):
        if m.get("role") == "user" and not user:
            user = str(m.get("content", ""))
        if m.get("role") == "system" and not system:
            system = str(m.get("content", ""))
    if response_format and response_format.get("type") == "json_object":
        if category == "judge":
            return json.dumps(
                {
                    "faithfulness": 0.86,
                    "answer_relevancy": 0.88,
                    "context_coverage": 0.8,
                    "unsupported_claims": [],
                }
            )
        if category == "main_agent":
            return json.dumps({"knowledge_base": "customer_support", "confidence": 0.7, "reason": "mock"})
        return json.dumps({"result": "mock", "query_excerpt": user[:80]})
    if category == "suggestion":
        return f"{user[:120]} exact section clause details"
    if category == "responsibility":
        low = user.lower()
        harmful = any(
            k in low for k in (
                "subliminal", "manipulate", "inferior", "subhuman", "should be excluded",
                "should be banned", "social scoring", "social credit", "penalize female",
                "penalize male", "deduct pay", "mass surveillance", "one ethnic group",
                "one race", "hate", "slur", "exterminat", "get rid of",
            )
        )
        if harmful:
            return (
                "### 1. COMPLIANCE VERDICT\n- **STATUS**: UNETHICAL / NON-COMPLIANT (FLAGGED)\n\n"
                "### 3. LEGAL & FRAMEWORK ANALYSIS\n"
                "- **EU AI Act, Article 5**: prohibited manipulative / discriminatory practice.\n"
                "- **NIST AI RMF**: violates the 'Fair - with harmful bias managed' characteristic.\n"
            )
        return (
            "### 1. COMPLIANCE VERDICT\n- **STATUS**: COMPLIANT\n\n"
            "### 3. LEGAL & FRAMEWORK ANALYSIS\n- (mock) no violation identified.\n"
        )
    # answer generation: echo whole sentences from the retrieved context so downstream
    # eval (entity drift / NLI) sees a genuinely grounded answer, not a mid-sentence cut
    if category in {"medium", "heavy", "light"} and "Context:" in system:
        ctx = " ".join(system.split("Context:", 1)[1].split())
        if ctx and ctx != "(no context retrieved)":
            sents = re.split(r"(?<=[.!?])\s+", ctx)
            picked, total = [], 0
            for s in sents:
                if total + len(s) > 360 and picked:
                    break
                picked.append(s)
                total += len(s)
            return "Based on the knowledge base: " + " ".join(picked)
    return f"Based on the knowledge base: {user[:200]}" if user else "no input"


class _MockToolCall:
    def __init__(self, name: str):
        self.function = type("F", (), {"name": name, "arguments": "{}"})()
        self.id = "mock_tc"
        self.type = "function"


class _MockRaw:
    def __init__(self, text: str, tool_name: Optional[str]):
        msg = type("M", (), {"content": text, "tool_calls": [_MockToolCall(tool_name)] if tool_name else None})()
        self.choices = [type("C", (), {"message": msg})()]
        self.model = "mock"
        self._hidden_params = {}


def _mock_tool_name(messages, tools) -> Optional[str]:
    names = [t.get("function", {}).get("name", "") for t in (tools or [])]
    user = next((str(m.get("content", "")) for m in reversed(messages) if m.get("role") == "user"), "").lower()
    kws = {
        "retrieve_customer_support": ["refund", "order", "cancel", "shipping", "return", "password", "account", "delivery", "payment"],
        "retrieve_hr_policy": ["leave", "salary", "resignation", "attendance", "dress code", "probation", "kespl", "notice period"],
        "retrieve_internal_knowledge": ["azure", "app service", "vnet", "tls", "deployment", "custom domain", "cli", "slot"],
        "retrieve_toxicity_kb": ["toxic", "hate speech", "stereotype", "offensive", "slur", "inferior", "subhuman"],
        "retrieve_decision_support": ["meeting", "remote control", "lcd", "led", "prototype", "target cost", "marketing", "demographic"],
    }
    best, hi = None, 0
    for name in names:
        n = sum(1 for k in kws.get(name, []) if k in user)
        if n > hi:
            best, hi = name, n
    return best or (names[0] if names else None)


def _mock_result(category: str, messages, response_format, stream: bool, tools=None):
    text = _mock_text(category, messages, response_format)
    delay_ms = float(os.getenv("CP_MOCK_LLM_MS", "0") or 0)
    if delay_ms and not stream:
        time.sleep(delay_ms / 1000.0)
    tool_name = _mock_tool_name(messages, tools) if tools else None
    res = LLMResult(
        text=text,
        model=f"mock/{category}",
        category=category,
        tier=1,
        prompt_tokens=sum(len(str(m.get("content", "")).split()) for m in messages),
        completion_tokens=len(text.split()),
        latency_ms=delay_ms or 5.0,
        cost_usd=0.0,
        provider="mock",
        mocked=True,
        raw=_MockRaw(text, tool_name),
    )
    if not stream:
        return res

    def _gen() -> Iterator[str]:
        for tok in re.findall(r"\S+\s*", text):
            time.sleep(0.002)
            yield tok

    return _gen(), res


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------
def _extract(resp, category: str, elapsed_ms: float) -> LLMResult:
    import litellm

    text = ""
    try:
        msg = resp.choices[0].message
        text = (msg.content or getattr(msg, "reasoning_content", "") or getattr(msg, "reasoning", "") or "")
    except Exception:
        text = str(resp)
    text = _strip_think(text)

    hp = getattr(resp, "_hidden_params", {}) or {}
    model_id = _real_model_name(hp.get("model_name"), hp.get("model"),
                                getattr(resp, "model", None), category)
    provider = _provider_of(str(model_id))

    usage = getattr(resp, "usage", None)
    pt = getattr(usage, "prompt_tokens", 0) or 0
    ct = getattr(usage, "completion_tokens", 0) or 0

    cost = 0.0
    if provider not in {"groq", "mock"}:
        try:
            cost = float(litellm.completion_cost(completion_response=resp) or 0.0)
        except Exception:
            cost = 0.0

    return LLMResult(
        text=text,
        model=str(model_id),
        category=category,
        tier=model_tier(str(model_id)),
        prompt_tokens=pt,
        completion_tokens=ct,
        latency_ms=elapsed_ms,
        cost_usd=cost,
        provider=provider,
        raw=resp,
    )


def complete(
    category: str,
    messages: List[Dict[str, str]],
    *,
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
    tools: Optional[List[dict]] = None,
    tool_choice: Optional[Any] = None,
    response_format: Optional[dict] = None,
) -> LLMResult:
    """Blocking chat completion. Never raises for provider errors that LiteLLM can retry;
    raises LLMUnavailable only if nothing is usable and we are not in mock mode."""
    if _is_mock():
        return _mock_result(category, messages, response_format, stream=False, tools=tools)

    router = get_router()
    if router is None:
        raise LLMUnavailable("No LiteLLM router (no API keys configured).")

    kwargs: Dict[str, Any] = {
        "model": category,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens or settings.request_max_tokens,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice or "auto"
    if response_format:
        kwargs["response_format"] = response_format

    t0 = time.perf_counter()
    resp = router.completion(**kwargs)
    elapsed = (time.perf_counter() - t0) * 1000.0
    result = _extract(resp, category, elapsed)

    # surface tool calls on the raw object; callers inspect result.raw
    return result


def stream_complete(
    category: str,
    messages: List[Dict[str, str]],
    *,
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
):
    """Yield token strings; the final item is an LLMResult with aggregated meta.

    Usage:
        gen = stream_complete(...)
        for piece in gen:
            if isinstance(piece, LLMResult): meta = piece
            else: print(piece, end="")
    """
    if _is_mock():
        gen, res = _mock_result(category, messages, None, stream=True)
        for tok in gen:
            yield tok
        yield res
        return

    router = get_router()
    if router is None:
        raise LLMUnavailable("No LiteLLM router (no API keys configured).")

    import litellm

    t0 = time.perf_counter()
    parts: List[str] = []            # visible answer content only
    # best-guess base model for the category (overridden by the real name from chunks)
    _cat_models = settings.model_catalog.get(category, [category])
    model_id = _cat_models[0] if _cat_models else category
    pt = ct = 0
    stream_failed = False
    buf = ""
    in_think = None
    saw_content = False

    try:
        chunks = router.completion(
            model=category,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens or settings.request_max_tokens,
            stream=True,
        )
        it = iter(chunks)
        # hard wall-clock cap on the streamed answer so a slow / run-away generation
        # can never blow the <10s pipeline budget - keep whatever arrived by then.
        _stream_cap = float(os.getenv("CP_STREAM_CAP_S", "7.0"))
        while True:
            if (time.perf_counter() - t0) > _stream_cap and parts:
                break
            try:
                chunk = next(it)
            except StopIteration:
                break
            except Exception:
                # Groq/LiteLLM occasionally emits a chunk with an empty `choices`
                # list that the parser trips over - stop cleanly and keep what we have.
                break
            try:
                choices = getattr(chunk, "choices", None) or []
                d = choices[0].delta if choices else None
                # ONLY the real `content` is streamed to the UI. `reasoning` /
                # `reasoning_content` (the model's chain-of-thought) is discarded -
                # streaming it made the UI show thinking text then jump to the answer.
                delta = getattr(d, "content", None) if d else None
            except Exception:
                delta = None
            if delta:
                saw_content = True
                parts.append(delta)
                if in_think is False:
                    yield delta
                else:
                    buf += delta
                    if in_think is None and len(buf) >= 7:
                        in_think = buf.lstrip().lower().startswith("<think>")
                        if not in_think:
                            yield buf
                            buf = ""
                    if in_think and "</think>" in buf:
                        in_think = False
                        tail = buf.split("</think>", 1)[1].lstrip()
                        if tail:
                            yield tail
                        buf = ""
            cm = getattr(chunk, "model", None)
            if cm and _HASH_RE.match(str(cm)) is None:
                model_id = str(cm)
            hp = getattr(chunk, "_hidden_params", {}) or {}
            if hp.get("model_name"):
                model_id = str(hp["model_name"])
            usage = getattr(chunk, "usage", None)
            if usage:
                pt = getattr(usage, "prompt_tokens", pt) or pt
                ct = getattr(usage, "completion_tokens", ct) or ct
    except Exception:
        stream_failed = True

    text = _strip_think("".join(parts))

    # nothing usable from streaming (or only <think> that got stripped) -> one non-stream call
    if not text.strip():
        try:
            res = complete(category, messages, temperature=temperature, max_tokens=max_tokens)
            if res.text:
                yield res.text
            yield res
            return
        except Exception as e:  # noqa: BLE001
            raise LLMUnavailable(f"streaming and fallback both failed: {e}")

    # flush any buffered-but-unyielded tail (stream ended before </think>, etc.)
    if buf.strip():
        t = _strip_think(buf)
        if t:
            yield t
    elapsed = (time.perf_counter() - t0) * 1000.0
    if not ct:
        ct = max(1, len(text.split()))
    provider = _provider_of(str(model_id))
    cost = 0.0
    if provider not in {"groq", "mock"}:
        try:
            cost = float(litellm.completion_cost(model=str(model_id), prompt="", completion=text) or 0.0)
        except Exception:
            cost = 0.0

    yield LLMResult(
        text=text, model=str(model_id), category=category, tier=model_tier(str(model_id)),
        prompt_tokens=pt, completion_tokens=ct, latency_ms=elapsed, cost_usd=cost, provider=provider,
    )


def complete_json(
    category: str,
    messages: List[Dict[str, str]],
    *,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    fallback: Optional[dict] = None,
) -> tuple[dict, LLMResult]:
    """Chat completion constrained to a JSON object. Returns (parsed_dict, meta).
    Robust to models that wrap JSON in prose or ```json fences."""
    res = complete(
        category,
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    parsed = _loads_lenient(res.text)
    if parsed is None:
        parsed = dict(fallback or {})
        parsed["_parse_error"] = True
    return parsed, res


def _loads_lenient(text: str) -> Optional[dict]:
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    # strip <think> traces (qwen-style) and code fences, then grab the first {...}
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"```(?:json)?", "", text)
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None
