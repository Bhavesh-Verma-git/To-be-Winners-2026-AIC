import os
from pathlib import Path
from dataclasses import dataclass

# Try to load python-dotenv, fallback to built-in parser if not installed
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Built-in .env parser fallback
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if env_file.exists():
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    if k and k not in os.environ:
                        os.environ[k] = v

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "Dataset"

NIST_PDF_PATH = DATASET_DIR / "NIST.AI.100-1.pdf"
EU_AI_ACT_PDF_PATH = DATASET_DIR / "OJ%3AL_202401689%3AEN%3ATXT.pdf"

CONTENT_MODERATION_PDFS = [
    (DATASET_DIR / "UN_Hate_Speech_Strategy.pdf", "UN Strategy and Plan of Action on Hate Speech", "un_hate_speech"),
    (DATASET_DIR / "UNESCO_Countering_Online_Hate_Speech.pdf", "UNESCO: Countering Online Hate Speech", "unesco_hate_speech"),
    (DATASET_DIR / "EU_Digital_Services_Act.pdf", "EU Digital Services Act (DSA)", "eu_dsa"),
    (DATASET_DIR / "EEOC_Harassment_Guidelines.pdf", "EEOC: Harassment in the Workplace Guidelines", "eeoc_harassment"),
    (DATASET_DIR / "CoE_Combating_Sexism.pdf", "Council of Europe - Preventing and Combating Sexism", "coe_sexism"),
    (DATASET_DIR / "OHCHR_Rabat_Plan.pdf", "OHCHR: Rabat Plan of Action", "ohchr_rabat")
]

@dataclass
class Settings:
    # -------------------------------------------------------------------------
    # 1. Final Responsibility Moderation Agent LLM (Google Gemini)
    # -------------------------------------------------------------------------
    AGENT_LLM_PROVIDER: str = os.getenv("AGENT_LLM_PROVIDER", "gemini").lower()
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    # -------------------------------------------------------------------------
    # 2. Knowledge Graph LLMGraphTransformer (Groq - Qwen)
    # -------------------------------------------------------------------------
    GRAPH_LLM_PROVIDER: str = os.getenv("GRAPH_LLM_PROVIDER", "groq").lower()
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_GRAPH_MODEL: str = os.getenv("GROQ_GRAPH_MODEL", "qwen/qwen3.6-27b")
    GROQ_MODERATION_MODEL: str = os.getenv("GROQ_MODERATION_MODEL", "qwen/qwen3.8-27b")
    
    # -------------------------------------------------------------------------
    # 3. Optional OpenAI Fallback Settings
    # -------------------------------------------------------------------------
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    
    # -------------------------------------------------------------------------
    # 4. Neo4j Graph Database Configuration
    # -------------------------------------------------------------------------
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USERNAME: str = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "password")
    NEO4J_DATABASE: str = os.getenv("NEO4J_DATABASE", "neo4j")
    
    # -------------------------------------------------------------------------
    # 5. Storage & Cache Paths
    # -------------------------------------------------------------------------
    DATA_DIR: Path = BASE_DIR / os.getenv("DATA_DIR", "data")
    VECTOR_DB_DIR: Path = BASE_DIR / os.getenv("VECTOR_DB_DIR", "data/chroma_db")
    CHUNK_STORE_PATH: Path = BASE_DIR / os.getenv("CHUNK_STORE_PATH", "data/chunk_store.json")
    BM25_STORE_PATH: Path = BASE_DIR / os.getenv("BM25_STORE_PATH", "data/bm25_index.pkl")
    GRAPH_CACHE_PATH: Path = BASE_DIR / os.getenv("GRAPH_CACHE_PATH", "data/graph_triples.json")
    
    # -------------------------------------------------------------------------
    # 6. Retrieval & Fusion Parameters
    # -------------------------------------------------------------------------
    RETRIEVAL_TOP_K_PER_BRANCH: int = int(os.getenv("RETRIEVAL_TOP_K_PER_BRANCH", "4"))
    RRF_TOP_K: int = int(os.getenv("RRF_TOP_K", "5"))
    RRF_K_CONSTANT: int = int(os.getenv("RRF_K_CONSTANT", "60"))

settings = Settings()

def ensure_data_directories():
    """Ensure that data and storage directories exist."""
    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
    settings.VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
