"""Semantic caching layer using Qdrant."""

from __future__ import annotations

import uuid
from datetime import datetime

from qdrant_client import models

from src.core import config
from src.core.embeddings import embed_query
from src.core.logging import get_logger
from src.rag.vector_store import get_qdrant_client

logger = get_logger(__name__)


class SemanticCache:
    """Qdrant-backed semantic cache for generation responses."""

    def __init__(self) -> None:
        """Initialize the Qdrant cache collection."""
        self.client = get_qdrant_client()
        self.collection_name = config.QDRANT_CACHE_COLLECTION

        if not self.client.collection_exists(collection_name=self.collection_name):
            logger.info("Creating Qdrant cache collection: %s", self.collection_name)
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=config.QDRANT_VECTOR_SIZE,
                    distance=models.Distance.COSINE,
                ),
            )

    def lookup(self, query_text: str) -> str | None:
        """
        Check if a semantically similar query exists in the cache.
        Returns the cached response_text if found (score >= threshold), else None.
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
        """
        Store a new query and response in the semantic cache.
        """
        query_embedding = embed_query(query_text)
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, query_text))

        payload = {
            "query": query_text,
            "response_text": response_text,
            "stored_at": datetime.utcnow().isoformat(),
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
