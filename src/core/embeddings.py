"""
core/embeddings.py

Centralized embedding client for the SEC compliance RAG system.

Responsibilities:
  - Provide local embedding capabilities for query and document chunks.
  - Abstract BAAI/bge-m3 model initialization and thread-safe lazy loading.

Not responsible for:
  - Chunking or text splitting (see rag/ingest.py).
  - Vector storage or retrieval (see rag/vector_store.py).

Dependencies:
  - FlagEmbedding >= 1.2
  - BGE_M3_MODEL_NAME and EMBEDDING_BATCH_SIZE must be set in config.
"""

from __future__ import annotations

from FlagEmbedding import BGEM3FlagModel

from src.core import config
from src.core.lazy import locked_singleton
from src.core.logging import get_logger

logger = get_logger(__name__)


@locked_singleton
def _get_model() -> BGEM3FlagModel:
    """Return the singleton BGE-M3 model, initializing it on first call.

    Uses a threading lock to prevent duplicate initialization when multiple
    threads request embeddings simultaneously during application startup.

    Returns:
        The initialized BGEM3FlagModel instance.
    """
    logger.info("Loading BGE-M3 embedding model — first call will download weights (~2.3 GB).")
    return BGEM3FlagModel(
        config.BGE_M3_MODEL_NAME,
        use_fp16=True,
        normalize_embeddings=True,
    )


def embed_query(text: str) -> list[float]:
    """Encode a single query string into a dense embedding vector.

    Uses a batch size of 1 and a conservative max_length of 512 tokens,
    appropriate for short user queries rather than long documents.

    Args:
        text: The raw query string to embed.

    Returns:
        A list of floats representing the normalized dense embedding vector.
    """
    model = _get_model()
    embeddings = model.encode(
        [text],
        batch_size=1,
        max_length=512,
        return_dense=True,
    )
    return embeddings["dense_vecs"][0].tolist()


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Encode a batch of document strings into dense embedding vectors.

    Uses an 8192-token max_length to capture full section-length chunks.
    Batch size is controlled by config.EMBEDDING_BATCH_SIZE to avoid
    GPU/CPU OOM during large ingestion runs.

    Args:
        texts: A list of document text strings to embed.

    Returns:
        A list of embedding vectors, one per input text, in the same order.
    """
    model = _get_model()
    embeddings = model.encode(
        texts,
        batch_size=config.EMBEDDING_BATCH_SIZE,
        max_length=8192,
        return_dense=True,
    )
    return [vec.tolist() for vec in embeddings["dense_vecs"]]
