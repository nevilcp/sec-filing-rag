"""Semantic caching layer using Qdrant."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from qdrant_client import models

from src.core import config
from src.core.embeddings import embed_query
from src.core.logging import get_logger
from src.rag.vector_store import ensure_collection, get_qdrant_client

logger = get_logger(__name__)


class SemanticCache:
    """Qdrant response cache keyed by query embeddings and matched by cosine similarity."""

    def __init__(self) -> None:
        self.client = get_qdrant_client()
        self.collection_name = config.QDRANT_CACHE_COLLECTION
        ensure_collection(
            self.client,
            self.collection_name,
            "All cached responses will be lost.\n         The cache will repopulate automatically on subsequent queries.",
        )

    def lookup(self, query_text: str, company_filter: str | None = None) -> dict[str, Any] | None:
        """Return the nearest cached answer if its similarity meets the configured threshold."""
        query_embedding = embed_query(query_text)

        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(key="cache_version", match=models.MatchValue(value=2)),
                    models.FieldCondition(
                        key="company_filter", match=models.MatchValue(value=(company_filter or "").strip().upper())
                    ),
                ]
            ),
            limit=1,
            with_payload=True,
        )

        if not search_result.points:
            return None

        best_hit = search_result.points[0]
        if best_hit.score >= config.CACHE_SIMILARITY_THRESHOLD:
            logger.debug("Semantic cache hit (score: %.3f)", best_hit.score)
            payload = best_hit.payload or {}
            return {
                "final_response": payload["response_text"],
                "retrieved_docs": payload.get("retrieved_docs", []),
                "query_category": payload.get("query_category", "general"),
                "retrieval_grade": payload.get("retrieval_grade", "relevant"),
            }

        logger.debug("Semantic cache miss (best score: %.3f)", best_hit.score)
        return None

    def store(
        self,
        query_text: str,
        response_text: str,
        *,
        company_filter: str | None = None,
        retrieved_docs: list[dict[str, Any]] | None = None,
        query_category: str = "general",
        retrieval_grade: str = "relevant",
    ) -> None:
        """Store an answer under a deterministic query ID, replacing any previous entry."""
        query_embedding = embed_query(query_text)
        company_filter = (company_filter or "").strip().upper()
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"v2:{company_filter}:{query_text}"))

        payload = {
            "query": query_text,
            "response_text": response_text,
            "cache_version": 2,
            "company_filter": company_filter,
            "retrieved_docs": retrieved_docs or [],
            "query_category": query_category,
            "retrieval_grade": retrieval_grade,
            "stored_at": datetime.now(UTC).isoformat(),
        }

        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=query_embedding,
                    payload=payload,
                )
            ],
        )
        logger.debug("Stored response in semantic cache.")
