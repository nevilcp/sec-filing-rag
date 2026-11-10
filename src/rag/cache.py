"""
rag/cache.py

Semantic caching layer using Qdrant.

Responsibilities:
  - Store and retrieve LLM generation responses based on query semantic similarity
  - Manage the initialization of the cache collection

Dependencies:
  - qdrant_client
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from qdrant_client import models

from src.core import config
from src.core.embeddings import embed_query
from src.core.logging import get_logger
from src.rag.vector_store import ensure_collection, get_qdrant_client

logger = get_logger(__name__)


class SemanticCache:
    """Qdrant-backed semantic cache for generation responses"""

    def __init__(self) -> None:
        """Initialize the semantic cache collection in Qdrant.

        If the collection already exists but has a mismatched vector size
        (e.g. after switching embedding models), the old collection is
        deleted and recreated, discarding all cached responses.
        """
        self.client = get_qdrant_client()
        self.collection_name = config.QDRANT_CACHE_COLLECTION
        ensure_collection(
            self.client,
            self.collection_name,
            "All cached responses will be lost.\n         The cache will repopulate automatically on subsequent queries.",
        )

    def lookup(self, query_text: str) -> str | None:
        """Check if a semantically similar query exists in the cache.

        Args:
            query_text: The user's raw query string.

        Returns:
            The cached response text if the cosine similarity is >= the configured
            threshold, otherwise None
        """
        query_embedding = embed_query(query_text)

        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=1,
            with_payload=True,
        )

        if not search_result.points:
            return None

        best_hit = search_result.points[0]
        if best_hit.score >= config.CACHE_SIMILARITY_THRESHOLD:
            logger.debug("Semantic cache hit (score: %.3f)", best_hit.score)
            payload = best_hit.payload or {}
            return payload.get("response_text")

        logger.debug("Semantic cache miss (best score: %.3f)", best_hit.score)
        return None

    def store(self, query_text: str, response_text: str) -> None:
        """Store a query-response pair in the semantic cache.

        Uses UUID5 on the raw query string to derive a deterministic point
        ID, so repeated stores for the same query overwrite rather than
        duplicate the cached entry.

        Args:
            query_text: The original user query string.
            response_text: The generated response to cache.
        """
        query_embedding = embed_query(query_text)
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, query_text))

        payload = {
            "query": query_text,
            "response_text": response_text,
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
