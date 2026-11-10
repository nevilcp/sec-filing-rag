"""Semantic caching layer using Qdrant.

Responsibilities:
  - Store generated answers keyed by the embedding of the query.
  - Return a stored answer for a sufficiently similar query.
  - Create the cache collection, recreating it if its vector size is
    stale.
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
    """Qdrant-backed semantic cache for generation responses.

    Creating an instance creates the cache collection if it is missing.
    A collection with a stale vector size, for example after an
    embedding model change, is deleted and recreated, which discards
    every cached response.

    Attributes:
        client: The shared local Qdrant client.
        collection_name: The Qdrant collection that holds the cache.
    """

    def __init__(self) -> None:
        self.client = get_qdrant_client()
        self.collection_name = config.QDRANT_CACHE_COLLECTION
        ensure_collection(
            self.client,
            self.collection_name,
            "All cached responses will be lost.\n         The cache will repopulate automatically on subsequent queries.",
        )

    def lookup(self, query_text: str) -> str | None:
        """Return the cached answer for a semantically similar query.

        Args:
            query_text: The user's raw query string.

        Returns:
            The cached response text if the best match has a cosine
            similarity of at least config.CACHE_SIMILARITY_THRESHOLD,
            otherwise None.
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
        """Store a query and response pair in the cache.

        The point ID is a UUID5 of the raw query, so storing the same
        query again overwrites its entry instead of adding a duplicate.

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
