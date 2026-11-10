"""Centralized embedding client for the SEC compliance RAG system.

Responsibilities:
  - Provide local embeddings for queries and document chunks.
  - Load the BAAI/bge-m3 model lazily and share one instance.

Not responsible for:
  - Chunking or text splitting (see rag/ingest.py).
  - Vector storage or retrieval (see rag/vector_store.py).

Dependencies:
  - The BGE-M3 weights, downloaded on first use (about 2.3 GB).
"""

from __future__ import annotations

from FlagEmbedding import BGEM3FlagModel

from src.core import config
from src.core.lazy import locked_singleton
from src.core.logging import get_logger

logger = get_logger(__name__)


@locked_singleton
def _get_model() -> BGEM3FlagModel:
    """Return the shared BGE-M3 model, loading it on first call."""
    logger.info("Loading BGE-M3 embedding model. The first call downloads weights (~2.3 GB).")
    return BGEM3FlagModel(
        config.BGE_M3_MODEL_NAME,
        use_fp16=True,
        normalize_embeddings=True,
    )


def embed_query(text: str) -> list[float]:
    """Encode a query string into a dense embedding vector.

    Uses a batch size of 1 and a max_length of 512 tokens, which suits
    short user queries.

    Args:
        text: The raw query string to embed.

    Returns:
        The normalized dense embedding vector.

    Example:
        >>> vector = embed_query("Apple net sales")
        >>> len(vector) == config.QDRANT_VECTOR_SIZE
        True
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
    """Encode document strings into dense embedding vectors.

    Uses a max_length of 8192 tokens to cover full section-length
    chunks. config.EMBEDDING_BATCH_SIZE sets the batch size, which
    keeps memory bounded during large ingestion runs.

    Args:
        texts: The document texts to embed.

    Returns:
        One embedding vector per input text, in input order.

    Example:
        >>> vectors = embed_documents(["chunk one", "chunk two"])
        >>> len(vectors)
        2
    """
    model = _get_model()
    embeddings = model.encode(
        texts,
        batch_size=config.EMBEDDING_BATCH_SIZE,
        max_length=8192,
        return_dense=True,
    )
    return [vec.tolist() for vec in embeddings["dense_vecs"]]
