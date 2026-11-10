"""
core/config.py

Centralized configuration for the SEC compliance RAG system.

Responsibilities:
  - Define filesystem paths for data, logs, and indexes.
  - Assign specific LLM models to LangGraph nodes based on
    performance and rate-limit requirements.
  - Configure retrieval thresholds and chunking parameters.

Not responsible for:
  - Runtime validation of env vars (see corpus.py and edgar_client.py).
  - Rate-limit enforcement (see core/rate_limiter.py).

Dependencies:
  - python-dotenv
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"
INDEX_DIR = DATA_DIR / "indexes"

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR_APP = LOG_DIR / "app"
LOG_DIR_API = LOG_DIR / "api"
LOG_DIR_RAG = LOG_DIR / "rag"
LOG_DIR_PHOENIX = LOG_DIR / "phoenix"
LOG_DIR_SYSTEM = LOG_DIR / "system"
LOG_DIR_AUDIT = LOG_DIR / "audit"

LOG_ROTATION_WHEN = "midnight"
LOG_ROTATION_BACKUP_COUNT = 14


INPUT_GUARD_MODEL = "meta-llama/llama-prompt-guard-2-86m"
OUTPUT_GUARD_MODEL = "openai/gpt-oss-safeguard-20b"
# Prompt Guard 2 returns a probability; scores at or above this block.
INPUT_GUARD_INJECTION_THRESHOLD = 0.5
ROUTER_MODEL = "openai/gpt-oss-20b"
# Grader and rewriter use a different model from the router so each
# role gets its own Groq per-model TPM budget on the free tier
GRADER_MODEL = "openai/gpt-oss-120b"
REWRITER_MODEL = "openai/gpt-oss-120b"
GENERATION_MODEL = "gemini-3.5-flash"
# Judge model for the Phoenix evals, reached through the Groq
# OpenAI-compatible endpoint because litellm is not installed.
EVAL_MODEL = "openai/gpt-oss-120b"
EVAL_BASE_URL = "https://api.groq.com/openai/v1"


GENERATION_LLM_TEMPERATURE = 0.1
ROUTING_LLM_TEMPERATURE = 0.0
GRADER_LLM_TEMPERATURE = 0.0
REWRITER_LLM_TEMPERATURE = 0.3


QDRANT_PATH = str(VECTORSTORE_DIR / "qdrant")
QDRANT_COLLECTION_NAME = "edgar_filings"
QDRANT_CACHE_COLLECTION = "semantic_cache"
QDRANT_VECTOR_SIZE = 1024
BGE_M3_MODEL_NAME = "BAAI/bge-m3"
BM25_INDEX_PATH = str(INDEX_DIR / "bm25s_index")
INGEST_CHECKPOINT_PATH = str(DATA_DIR / ".ingest_checkpoint.json")
EDGAR_STAGING_PATH = str(DATA_DIR / "edgar_staging")
CIK_CACHE_PATH = str(DATA_DIR / "company_tickers.json")


CHUNK_SIZE = 750
CHUNK_OVERLAP = 150


CACHE_SIMILARITY_THRESHOLD = 0.92
TOP_K_RETRIEVAL_COUNT = 5
BGE_SCORE_THRESHOLD = 0.25
RELEVANCE_MEAN_THRESHOLD = 0.50
RELEVANCE_TOP1_THRESHOLD = 0.70


MAX_CRAG_REWRITES = 2


EDGAR_RATE_LIMIT_REQUESTS_PER_SECOND = 7


EMBEDDING_BATCH_SIZE = 5
EDGAR_TARGET_SECTIONS = ["Item 1", "Item 1A", "Item 1C", "Item 3", "Item 7", "Item 7A"]
EDGAR_TARGET_FORM_TYPES = ["10-K"]
_years_env = os.getenv("EDGAR_FISCAL_YEAR")
if not _years_env:
    raise ValueError(
        "CRITICAL: 'EDGAR_FISCAL_YEAR' environment variable is missing. Specify one or more comma-separated years."
    )
EDGAR_FISCAL_YEARS = [y.strip() for y in _years_env.split(",") if y.strip()]


RETRY_MULTIPLIER_SECONDS = 2.0
RETRY_MIN_WAIT_SECONDS = 5.0
RETRY_MAX_WAIT_SECONDS = 60.0
RETRY_MAX_ATTEMPTS = 8
# SDK-level retries cover transient 5xx and connection errors. Tenacity
# handles 429s, so a larger value here multiplies the call count.
LLM_SDK_MAX_RETRIES = 2


API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))


PHOENIX_HOST = os.getenv("PHOENIX_HOST", "localhost")
PHOENIX_PORT = int(os.getenv("PHOENIX_PORT", "6006"))
# Project that run.py registers traces under; the evals read it back.
PHOENIX_PROJECT_NAME = "sec-filing-rag"
