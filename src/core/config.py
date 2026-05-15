"""Centralized configuration constants for the SEC compliance RAG system."""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
import os
os.environ["CHROMA_TELEMETRY"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "False"


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"
INDEX_DIR = DATA_DIR / "indexes"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "models/gemini-embedding-001"
GENERATION_PRIMARY_MODEL = "gemini-3.1-flash-lite"   # Tier 1: 500 RPD
GENERATION_PREMIUM_MODEL = "gemini-3-flash-preview"         # Tier 2: 20 RPD
CLASSIFICATION_MODEL = "gemini-3.1-flash-lite"    # Tier 1: guardrails
GENERATION_LLM_TEMPERATURE = 0.1
ROUTING_LLM_TEMPERATURE = 0.3


# ---------------------------------------------------------------------------
# Storage/index paths
# ---------------------------------------------------------------------------
QDRANT_PATH = str(VECTORSTORE_DIR / "qdrant")
QDRANT_COLLECTION_NAME = "edgar_filings"
QDRANT_CACHE_COLLECTION = "semantic_cache"
QDRANT_VECTOR_SIZE = 3072   # Gemini Embedding 2 output dimension
BM25_INDEX_PATH = str(INDEX_DIR / "bm25s_index")   # directory, not file
INGEST_CHECKPOINT_PATH = str(DATA_DIR / ".ingest_checkpoint.json")
EDGAR_STAGING_PATH = str(DATA_DIR / "edgar_staging")
CORPUS_MANIFEST_PATH = str(DATA_DIR / "corpus_manifest.json")
CIK_CACHE_PATH = str(DATA_DIR / "company_tickers.json")


# ---------------------------------------------------------------------------
# Chunking and filtering
# ---------------------------------------------------------------------------
CHUNK_SIZE = 750
CHUNK_OVERLAP = 150


# ---------------------------------------------------------------------------
# Retrieval and ranking
# ---------------------------------------------------------------------------
CACHE_SIMILARITY_THRESHOLD = 0.92
TOP_K_RETRIEVAL_COUNT = 8
BGE_SCORE_THRESHOLD = 0.25


# ---------------------------------------------------------------------------
# Iteration limits
# ---------------------------------------------------------------------------
MAX_CRAG_REWRITE_ITERATIONS = 2
MAX_CRAG_REWRITES = MAX_CRAG_REWRITE_ITERATIONS
MAX_REACT_ITERATIONS = 6


# ---------------------------------------------------------------------------
# Rate limiting parameters
# ---------------------------------------------------------------------------
API_RATE_LIMIT_REQUESTS_PER_MINUTE = 120
API_RATE_LIMIT_BURST = 30
LLM_RATE_LIMIT_REQUESTS_PER_MINUTE = 60
LLM_RATE_LIMIT_TOKENS_PER_MINUTE = 120_000
EDGAR_RATE_LIMIT_REQUESTS_PER_SECOND = 7   # Conservative below 10 req/sec hard cap

# Gemini Embedding 2 API quotas (from API dashboard)
# Conservative margins to avoid server-side window collisions from prior runs
EMBEDDING_RATE_LIMIT_REQUESTS_PER_MINUTE = 50   # Hard cap: 100 RPM
EMBEDDING_TOKENS_PER_MINUTE = 20_000             # Hard cap: 30K TPM
EMBEDDING_REQUESTS_PER_DAY = 900                  # Hard cap: 1K RPD
EMBEDDING_MIN_REQUEST_INTERVAL = 1.5              # Seconds between requests (prevents bursting)


# ---------------------------------------------------------------------------
# Ingestion execution parameters
# ---------------------------------------------------------------------------
EMBEDDING_BATCH_SIZE = 7  # Conservative to stay within TPM ceiling
EDGAR_TARGET_SECTIONS = ["Item 1", "Item 1A", "Item 7", "Item 7A"]
EDGAR_TARGET_FORM_TYPES = ["10-K"]


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------
PHOENIX_PORT = 6006
PHOENIX_HOST = "localhost"


# ---------------------------------------------------------------------------
# Retry policy (rate limit handling)
# ---------------------------------------------------------------------------
RETRY_MULTIPLIER_SECONDS = 2.0
RETRY_MIN_WAIT_SECONDS = 5.0
RETRY_MAX_WAIT_SECONDS = 60.0
RETRY_MAX_ATTEMPTS = 8
