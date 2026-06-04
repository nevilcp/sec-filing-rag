"""
rag/vector_store.py

Qdrant collection setup and dense retrieval.

Responsibilities:
  - Maintain Qdrant local client lifecycle and configuration
  - Ensure collection schemas align with embedded vector dimensions
  - Execute dense vector similarity searches with optional payload filtering
  - Upsert document chunks efficiently

Dependencies:
  - qdrant_client
"""

from __future__ import annotations

import uuid
from typing import Any

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from src.core import config
from src.core.embeddings import embed_query
from src.core.logging import get_logger

logger = get_logger(__name__)

_QDRANT_CLIENT: QdrantClient | None = None
_COLLECTION_CHECKED: bool = False


def get_qdrant_client() -> QdrantClient:
    """Return the singleton local Qdrant client, creating it on first call.

    The client is stored in a module-level global so repeated calls within
    the same process reuse the same connection to the local file-based store.

    Returns:
        The initialized QdrantClient instance.
    """
    global _QDRANT_CLIENT
    if _QDRANT_CLIENT is None:
        _QDRANT_CLIENT = QdrantClient(path=config.QDRANT_PATH)
    return _QDRANT_CLIENT


def get_filings_collection() -> None:
    """Ensure the primary filings collection exists in Qdrant.

    Idempotent: skips collection checks after the first successful
    verification within the process lifetime. If the existing collection
    has a mismatched vector size (e.g. after a model change), it is
    deleted and recreated, discarding all indexed documents.

    Raises:
        qdrant_client.http.exceptions.UnexpectedResponse: On unexpected
            Qdrant API errors during creation.
    """
    global _COLLECTION_CHECKED
    if _COLLECTION_CHECKED:
        return

    client = get_qdrant_client()
    try:
        collection_info = client.get_collection(config.QDRANT_COLLECTION_NAME)
        existing_size = collection_info.config.params.vectors.size
        if existing_size != config.QDRANT_VECTOR_SIZE:
            logger.warning(
                "Existing Qdrant collection has vector size %d but config requires %d.\n"
                "         Deleting and recreating — all indexed documents will be lost.\n"
                "         Re-run `ingest` to rebuild the corpus.",
                existing_size,
                config.QDRANT_VECTOR_SIZE,
            )
            client.delete_collection(config.QDRANT_COLLECTION_NAME)
            client.create_collection(
                collection_name=config.QDRANT_COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=config.QDRANT_VECTOR_SIZE,
                    distance=models.Distance.COSINE,
                ),
            )
    except (UnexpectedResponse, ValueError):
        client.create_collection(
            collection_name=config.QDRANT_COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=config.QDRANT_VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
        )
    _COLLECTION_CHECKED = True


def dense_search(
    query_text: str,
    top_k: int,
    filter_ticker: str | None = None,
    filter_section: str | None = None,
    filter_form_type: str | None = None,
) -> list[dict[str, Any]]:
    """Runs dense vector search in Qdrant and returns normalized result dicts.

    Args:
        query_text: The user's query.
        top_k: Maximum number of results to return.
        filter_ticker: Optional metadata filter for ticker.
        filter_section: Optional metadata filter for section name.
        filter_form_type: Optional metadata filter for form type.

    Returns:
        A list of search results containing text, score, and metadata
    """
    client = get_qdrant_client()
    get_filings_collection()

    query_embedding = embed_query(query_text)

    must_conditions = []
    if filter_ticker:
        must_conditions.append(
            models.FieldCondition(
                key="ticker",
                match=models.MatchValue(value=filter_ticker),
            )
        )
    if filter_section:
        must_conditions.append(
            models.FieldCondition(
                key="section_name",
                match=models.MatchValue(value=filter_section),
            )
        )
    if filter_form_type:
        must_conditions.append(
            models.FieldCondition(
                key="form_type",
                match=models.MatchValue(value=filter_form_type),
            )
        )

    query_filter = models.Filter(must=must_conditions) if must_conditions else None

    search_result = client.query_points(
        collection_name=config.QDRANT_COLLECTION_NAME,
        query=query_embedding,
        query_filter=query_filter,
        limit=top_k,
        with_payload=True,
    )

    formatted: list[dict[str, Any]] = []
    for hit in search_result.points:
        payload = dict(hit.payload) if hit.payload else {}
        text = payload.pop("text", "")
        formatted.append(
            {
                "text": text,
                "score": hit.score,
                "metadata": payload,
            }
        )
    return formatted


def upsert_chunks(chunks: list[dict[str, Any]]) -> None:
    """Upsert a batch of embedded document chunks into the Qdrant collection.

    Uses UUID5 (deterministic, namespace-DNS) to derive point IDs from
    the chunk's composite 'id' field, making upserts idempotent: re-running
    ingestion for the same accession number overwrites existing points
    rather than creating duplicates.

    Args:
        chunks: A list of chunk dicts, each containing 'id', 'text',
            'metadata', and 'embedding' keys.
    """
    client = get_qdrant_client()
    get_filings_collection()

    points = []
    for chunk in chunks:
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk["id"]))
        payload = chunk["metadata"].copy()
        payload["text"] = chunk["text"]

        points.append(
            models.PointStruct(
                id=point_id,
                vector=chunk["embedding"],
                payload=payload,
            )
        )

    if points:
        client.upsert(
            collection_name=config.QDRANT_COLLECTION_NAME,
            points=points,
        )
