"""Centralized embedding client for the SEC compliance RAG system."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core import config
from src.core.lazy import locked_singleton
from src.core.logging import get_logger

if TYPE_CHECKING:
    from FlagEmbedding import BGEM3FlagModel

logger = get_logger(__name__)


@locked_singleton
def _get_model() -> BGEM3FlagModel:
    from FlagEmbedding import BGEM3FlagModel

    logger.info("Loading BGE-M3 embedding model. The first call downloads weights (~2.3 GB).")
    return BGEM3FlagModel(
        config.BGE_M3_MODEL_NAME,
        use_fp16=True,
        normalize_embeddings=True,
    )


def embed_query(text: str) -> list[float]:
    """Encode a query as a normalized dense vector with a 512-token limit."""
    model = _get_model()
    embeddings = model.encode(
        [text],
        batch_size=1,
        max_length=512,
        return_dense=True,
    )
    return embeddings["dense_vecs"][0].tolist()


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Encode documents in input order with an 8192-token limit per document."""
    model = _get_model()
    embeddings = model.encode(
        texts,
        batch_size=config.EMBEDDING_BATCH_SIZE,
        max_length=8192,
        return_dense=True,
    )
    return [vec.tolist() for vec in embeddings["dense_vecs"]]
