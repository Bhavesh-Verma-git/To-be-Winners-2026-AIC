"""
Central configuration for ControlPlane.ai.

- Loads environment from  <repo_root>/.env  and  controlplane/.env  (the latter wins).
- Parses the multi-key pools (GROQ_API_KEYS / GEMINI_API_KEYS, comma separated).
- Defines the LiteLLM MODEL_CATALOG (category -> ordered model list) with per-category
  env overrides (CP_MODEL_<CATEGORY>).
- Holds every tunable threshold and every path to an existing on-disk index so the
  rest of the package never hard-codes a location.

Nothing here imports heavy libraries - safe to import anywhere.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
PKG_DIR = Path(__file__).resolve().parent
REPO_ROOT = PKG_DIR.parent

RAG_AGENTS_DIR = REPO_ROOT / "rag_agents"
MASTER_ROUTER_DIR = REPO_ROOT / "master_router"
RESPONSIBILITY_DIR = REPO_ROOT / "Responsiblity Agent"


def _load_env() -> None:
    """Load .env files without an extra hard dependency (dotenv if present, else manual)."""
    candidates = [REPO_ROOT / ".env", PKG_DIR / ".env"]
    try:
        from dotenv import load_dotenv  # type: ignore

        for path in candidates:
            if path.exists():
                load_dotenv(path, override=True)
        return
    except Exception:
        pass

    for path in candidates:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            val = val.strip()
            # strip inline comments  (KEY=value   # note)  -> matches python-dotenv
            if not (val.startswith('"') or val.startswith("'")):
                val = val.split(" #", 1)[0].split("\t#", 1)[0].strip()
            os.environ[key.strip()] = val.strip().strip("'\"")


_load_env()

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")        # chromadb
os.environ.setdefault("CHROMA_TELEMETRY_ENABLED", "false")
os.environ.setdefault("POSTHOG_DISABLED", "1")
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")  # offline cost map
os.environ.setdefault("GRPC_VERBOSITY", "NONE")

# quiet the noisy third-party loggers (neo4j routing chatter, litellm info spam)
import logging as _logging  # noqa: E402

for _n in ("neo4j", "neo4j.pool", "neo4j.io", "neo4j.bolt", "neo4j._async_compat",
           "LiteLLM", "litellm", "httpx", "httpcore"):
    _lg = _logging.getLogger(_n)
    _lg.setLevel(_logging.CRITICAL)
    _lg.propagate = False


def _maybe_go_offline() -> None:
    """If every HF model we need is already cached, skip transformers' per-call
    online revision checks (a big latency win). Force online with CP_HF_ONLINE=1."""
    if os.getenv("CP_HF_ONLINE", "").lower() in {"1", "true", "yes"}:
        return
    if os.getenv("HF_HUB_OFFLINE"):
        return
    from pathlib import Path as _P

    hub = _P(os.path.expanduser("~/.cache/huggingface/hub"))
    needed = [
        "models--sentence-transformers--all-MiniLM-L6-v2",
        "models--BAAI--bge-small-en-v1.5",
        "models--roberta-large-mnli",
        "models--unitary--toxic-bert",
        "models--s-nlp--roberta_toxicity_classifier",
    ]
    if hub.exists() and all((hub / n).exists() for n in needed):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


_maybe_go_offline()


_PLACEHOLDER = {"", "your-key-here", "changeme"}


def _split_keys(*names: str, prefix: Optional[str] = None) -> List[str]:
    """Collect API keys from any of the given env vars (comma separated), de-duped, ordered.
    If `prefix` is given, only keys starting with it are kept (drops wrong-format keys
    that would otherwise fail slowly through LiteLLM)."""
    seen: List[str] = []
    for name in names:
        raw = os.getenv(name, "")
        for part in raw.replace(";", ",").split(","):
            part = part.strip()
            if not part or part.lower() in _PLACEHOLDER or part in seen:
                continue
            if prefix and not part.startswith(prefix):
                continue
            seen.append(part)
    return seen


def _gemini_keys() -> List[str]:
    """Google AI Studio API keys look like `AIza...`. OAuth access tokens
    (`AQ.` / `ya29.`) are NOT accepted by LiteLLM's `gemini/` provider - they
    fail every call with a 401 `ACCESS_TOKEN_TYPE_UNSUPPORTED`, which just adds
    latency. Drop them unless CP_GEMINI_ALLOW_ANY=1 forces them back in."""
    keys = _split_keys("GEMINI_API_KEYS", "GOOGLE_API_KEY", "GEMINI_API_KEY")
    if os.getenv("CP_GEMINI_ALLOW_ANY", "").strip().lower() in {"1", "true", "yes"}:
        return keys
    return [k for k in keys if k.startswith("AIza")]


def _flag(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _f(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


def _i(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


def _valid_ls_key() -> bool:
    k = (os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY") or "").strip()
    return len(k) > 20 and (k.startswith("lsv2_") or k.startswith("ls__"))


# --------------------------------------------------------------------------------------
# Model catalog
# --------------------------------------------------------------------------------------
# category -> ordered list of *base* model ids (LiteLLM format, no api_key yet).
# The router (llm/router.py) expands each of these over every available key of the
# matching provider and wires cross-provider fallbacks.
# Models verified available on the provided Groq keys (Jan 2026):
#   openai/gpt-oss-120b, openai/gpt-oss-20b, compound-beta, qwen/qwen3.6-27b, qwen/qwen3.8-27b
# Gemini fallback: gemini-2.5-flash (works with the provided key).
# No llama models. Override any category with CP_MODEL_<CATEGORY>.
_GPT120 = "groq/openai/gpt-oss-120b"
_GPT20 = "groq/openai/gpt-oss-20b"
_COMPOUND = "groq/compound-beta"
_QWEN36 = "groq/qwen/qwen3.6-27b"
_QWEN38 = "groq/qwen/qwen3.8-27b"
_GEM_FLASH = "gemini/gemini-2.5-flash"


def _cat(*models: str) -> List[str]:
    out = list(models)
    if _gemini_keys():          # only add the Gemini fallback when a USABLE key exists
        out.append(_GEM_FLASH)
    return out


_DEFAULT_CATALOG: Dict[str, List[str]] = {
    # gpt-oss-* first everywhere (fast + reliable); qwen/compound as extra fallbacks.
    "light": _cat(_GPT20, _GPT120),
    "medium": _cat(_GPT120, _GPT20),
    "heavy": _cat(_GPT120, _GPT20, _COMPOUND, _QWEN38),   # decision-support reasoning
    "judge": _cat(_GPT120, _GPT20, _QWEN36),              # RAGAS faithfulness
    "main_agent": _cat(_GPT120, _GPT20),                  # router (tool calling)
    "suggestion": _cat(_GPT20, _GPT120),
    "responsibility": _cat(_GPT120, _GPT20, _QWEN38),     # violation report (only when flagged)
}


def _resolve_catalog() -> Dict[str, List[str]]:
    catalog: Dict[str, List[str]] = {}
    for category, models in _DEFAULT_CATALOG.items():
        override = os.getenv(f"CP_MODEL_{category.upper()}", "").strip()
        if override:
            catalog[category] = [m.strip() for m in override.split(",") if m.strip()]
        else:
            catalog[category] = list(models)
    return catalog


# Which model category answers each knowledge base (override with CP_KB_MODEL_<KB>).
_DEFAULT_KB_MODEL: Dict[str, str] = {
    "customer_support": "medium",
    "hr_policy": "medium",
    "internal_knowledge": "medium",
    "toxicity_kb": "medium",
    # meeting reasoning still uses gpt-oss-120b (medium's primary) but drops the
    # slow compound-beta / qwen reasoning fallbacks that pushed latency past 10s.
    "decision_support": "medium",   # set CP_KB_MODEL_DECISION_SUPPORT=heavy to restore
}


def _resolve_kb_model() -> Dict[str, str]:
    out = {}
    for kb, cat in _DEFAULT_KB_MODEL.items():
        out[kb] = os.getenv(f"CP_KB_MODEL_{kb.upper()}", cat).strip() or cat
    return out


@dataclass(frozen=True)
class Settings:
    # ---- API key pools -------------------------------------------------------------
    groq_keys: List[str] = field(
        default_factory=lambda: _split_keys("GROQ_API_KEYS", "GROQ_API_KEY", prefix="gsk_")
    )
    gemini_keys: List[str] = field(default_factory=_gemini_keys)

    # ---- LiteLLM catalog ----------------------------------------------------------
    model_catalog: Dict[str, List[str]] = field(default_factory=_resolve_catalog)
    kb_model: Dict[str, str] = field(default_factory=_resolve_kb_model)
    litellm_num_retries: int = field(default_factory=lambda: _i("CP_LITELLM_RETRIES", 1))
    litellm_timeout: float = field(default_factory=lambda: _f("CP_LITELLM_TIMEOUT", 12.0))
    request_max_tokens: int = field(default_factory=lambda: _i("CP_MAX_TOKENS", 1024))

    # ---- Guardrails -------------------------------------------------------------
    guard_jailbreak_similarity: float = field(default_factory=lambda: _f("CP_GUARD_SIM", 0.78))

    # ---- Semantic cache -------------------------------------------------------------
    cache_threshold: float = field(default_factory=lambda: _f("CP_CACHE_THRESHOLD", 0.80))
    cache_enabled: bool = field(default_factory=lambda: _flag("CP_CACHE_ENABLED", True))
    # in-memory by default: every app start begins with an EMPTY cache; within a
    # running session the 2nd similar query hits. Set CP_CACHE_PERSIST=1 for disk.
    cache_persist: bool = field(default_factory=lambda: _flag("CP_CACHE_PERSIST", False))
    cache_store_dir: Path = field(
        default_factory=lambda: Path(os.getenv("CP_CACHE_DIR", str(PKG_DIR / "cache" / "store")))
    )

    # ---- Retrieval -------------------------------------------------------------
    vector_top_k: int = field(default_factory=lambda: _i("CP_VECTOR_K", 5))
    bm25_top_k: int = field(default_factory=lambda: _i("CP_BM25_K", 5))
    rrf_top_k: int = field(default_factory=lambda: _i("CP_RRF_K", 5))
    rrf_k_constant: int = field(default_factory=lambda: _i("CP_RRF_C", 60))

    # ---- Performance branch -------------------------------------------------------------
    xgb_max_sentences: int = field(default_factory=lambda: _i("CP_XGB_MAX_SENTS", 0))
    # Default to a small fast NLI model to hold the <10s latency budget on CPU.
    # Set CP_NLI_MODEL=roberta-large-mnli for the training-faithful (slower) model.
    nli_model: str = field(default_factory=lambda: os.getenv("CP_NLI_MODEL", "cross-encoder/nli-distilroberta-base"))
    ragas_faithfulness_fail: float = field(default_factory=lambda: _f("CP_RAGAS_FAIL", 0.50))
    ragas_faithfulness_pass: float = field(default_factory=lambda: _f("CP_RAGAS_PASS", 0.70))
    xgb_hallucination_threshold: float = field(default_factory=lambda: _f("CP_XGB_THRESH", 0.60))

    # ---- Responsibility branch -------------------------------------------------------------
    tox_hard_threshold: float = field(default_factory=lambda: _f("CP_TOX_HARD", 0.70))
    tox_soft_threshold: float = field(default_factory=lambda: _f("CP_TOX_SOFT", 0.40))
    responsibility_top_k_per_branch: int = field(default_factory=lambda: _i("CP_RESP_BRANCH_K", 4))
    responsibility_rrf_top_k: int = field(default_factory=lambda: _i("CP_RESP_RRF_K", 5))
    detoxify_variant: str = field(default_factory=lambda: os.getenv("CP_DETOXIFY_VARIANT", "original"))

    # ---- Retry / HITL -------------------------------------------------------------
    max_hallucination_retries: int = 1
    max_hitl_rounds: int = 1

    # ---- Embeddings -------------------------------------------------------------
    minilm_model: str = field(default_factory=lambda: os.getenv("CP_MINILM_MODEL", "all-MiniLM-L6-v2"))
    bge_model: str = field(default_factory=lambda: os.getenv("CP_BGE_MODEL", "BAAI/bge-small-en-v1.5"))
    embed_device: str = field(default_factory=lambda: os.getenv("CP_EMBED_DEVICE", "cpu"))

    # ---- Neo4j -------------------------------------------------------------
    neo4j_uri: str = field(default_factory=lambda: os.getenv("NEO4J_URI", ""))
    neo4j_user: str = field(default_factory=lambda: os.getenv("NEO4J_USERNAME", "neo4j"))
    neo4j_password: str = field(default_factory=lambda: os.getenv("NEO4J_PASSWORD", ""))
    neo4j_database: str = field(default_factory=lambda: os.getenv("NEO4J_DATABASE", "neo4j"))

    # ---- Observability -------------------------------------------------------------
    # only "enabled" with a real-looking key - a placeholder makes LiteLLM's langsmith
    # callback throw on every call.
    langsmith_enabled: bool = field(default_factory=lambda: _valid_ls_key())
    langsmith_project: str = field(
        default_factory=lambda: os.getenv("LANGCHAIN_PROJECT", "controlplane")
    )

    # ---- Latency budget -------------------------------------------------------------
    latency_budget_s: float = field(default_factory=lambda: _f("CP_LATENCY_BUDGET_S", 10.0))

    # ---- Index paths (existing on-disk artefacts) --------------------------------
    @property
    def paths(self) -> Dict[str, Path]:
        rag = RAG_AGENTS_DIR
        return {
            "cs_faiss": rag / "customer_support" / "faiss_index",
            "cs_bm25": rag / "customer_support" / "faiss_index" / "bm25_index.pkl",
            "hr_faiss": rag / "hr_policy" / "faiss_index",
            "hr_parents": rag / "hr_policy" / "faiss_index" / "parent_store.json",
            "hr_bm25": rag / "hr_policy" / "faiss_index" / "bm25_parents.pkl",  # built by scripts/build_hr_bm25
            "ik_faiss": rag / "internal_knowledge" / "faiss_index",
            "ik_bm25": rag / "internal_knowledge" / "faiss_index" / "bm25_index.pkl",
            "ik_chunks": rag / "internal_knowledge" / "faiss_index" / "chunks.jsonl",
            "tox_faiss": rag / "Toxic_RAG" / "Toxic_RAG" / ".index_cache" / "faiss_index",
            "tox_bm25": rag / "Toxic_RAG" / "Toxic_RAG" / ".index_cache" / "bm25_retriever.pkl",
            "tox_dataset": rag / "Toxic_RAG" / "Toxic_RAG" / "Dataset" / "final_tox_Rag.csv",
            "ds_dir": rag / "Decision Support Rag" / "Decision Support Rag" / "Data",
            "xgb_model": MASTER_ROUTER_DIR
            / "performance_branch"
            / "hallucination_classifier"
            / "model"
            / "xgb_hallucination_model.json",
            "resp_chunks": RESPONSIBILITY_DIR / "data" / "chunk_store.json",
            "resp_bm25": RESPONSIBILITY_DIR / "data" / "bm25_index.pkl",
            "resp_triples": RESPONSIBILITY_DIR / "data" / "graph_triples.json",
            "resp_chroma_local": RESPONSIBILITY_DIR / "data" / "chroma_db_local",
            "resp_matrix": RESPONSIBILITY_DIR / "data" / "resp_minilm_matrix.npz",
        }

    def has_groq(self) -> bool:
        return len(self.groq_keys) > 0

    def has_gemini(self) -> bool:
        return len(self.gemini_keys) > 0

    def has_any_llm(self) -> bool:
        return self.has_groq() or self.has_gemini()


settings = Settings()

# Knowledge base identifiers used across the package (also the router tool names).
KB_IDS = [
    "customer_support",
    "hr_policy",
    "internal_knowledge",
    "toxicity_kb",
    "decision_support",
]

KB_LABELS = {
    "customer_support": "Customer Support",
    "hr_policy": "HR Policy (KESPL)",
    "internal_knowledge": "Azure App Service Docs",
    "toxicity_kb": "Toxicity / Content-Safety KB",
    "decision_support": "Decision Support (Meetings)",
    "none": "General / Out of scope",
}

KB_DESCRIPTIONS = {
    "customer_support": (
        "Customer service and e-commerce support: orders, refunds, cancellations, "
        "billing, shipping, delivery, account access, returns, subscriptions."
    ),
    "hr_policy": (
        "Kamaiah Engineering Services (KESPL) internal HR policy: leave (casual/sick/"
        "privilege), salary, promotion, attendance, dress code, discipline, termination, "
        "travel allowance, employee categories."
    ),
    "internal_knowledge": (
        "Microsoft Azure App Service technical documentation: deployment, scaling, custom "
        "domains, TLS/SSL, VNet integration, authentication, CLI commands, runtime config, "
        "diagnostics, staging slots."
    ),
    "toxicity_kb": (
        "Content-safety / hate-speech knowledge base. Contains a large annotated corpus of "
        "real toxic, hateful, offensive, discriminatory and stereotyping statements, jokes and "
        "opinions targeting demographic groups (race, ethnicity, nationality, religion, gender, "
        "sexual orientation, disability, age). Use this for ANY query about toxic / hateful / "
        "offensive / stereotypical / discriminatory views, jokes, slurs or statements about or "
        "targeting a group, questions asking what such views/stereotypes are, and requests to "
        "analyse, classify, rate or explain whether a statement is toxic or hate speech "
        "(target group, framing, stereotyping, factual, lewd labels)."
    ),
    "decision_support": (
        "Transcripts of corporate product-design meetings (remote-control product): team "
        "decisions, target costs, component choices (LCD vs LED, battery vs solar), "
        "demographics, marketing vs engineering positions."
    ),
}
