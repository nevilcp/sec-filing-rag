"""Qdrant collection setup and dense retrieval."""

from __future__ import annotations

import uuid
from typing import Any

from qdrant_client import QdrantClient, models

from src.core import config
from src.core.embeddings import embed_query

_QDRANT_CLIENT: QdrantClient | None = None
_COLLECTION_CHECKED: bool = False


def get_qdrant_client() -> QdrantClient:
    """Return cached Qdrant client."""
    global _QDRANT_CLIENT
    if _QDRANT_CLIENT is None:
        _QDRANT_CLIENT = QdrantClient(path=config.QDRANT_PATH)
    return _QDRANT_CLIENT


def get_filings_collection() -> None:
    """Ensures the Qdrant collection exists with correct vector config."""
    global _COLLECTION_CHECKED
    if _COLLECTION_CHECKED:
        return

    client = get_qdrant_client()
    if not client.collection_exists(collection_name=config.QDRANT_COLLECTION_NAME):
        client.create_collection(
            collection_name=config.QDRANT_COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=config.QDRANT_VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
        )
    _COLLECTION_CHECKED = True


# Backward-compatible alias for current ingestion imports
get_regulations_collection = get_filings_collection


def dense_search(
    query_text: str,
    top_k: int,
    filter_ticker: str | None = None,
    filter_section: str | None = None,
    filter_form_type: str | None = None,
) -> list[dict[str, Any]]:
    """Run dense vector search in Qdrant and return normalized result dicts."""
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
    """Upsert text chunks and their embeddings into Qdrant in a single batch call."""
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
