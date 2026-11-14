"""Cache round trips against a real local Qdrant collection."""

from unittest import mock

import pytest
from qdrant_client import QdrantClient, models

from src.core import config
from src.rag import cache


@pytest.fixture
def semantic_cache(monkeypatch):
    client = QdrantClient(location=":memory:")
    monkeypatch.setattr(config, "QDRANT_VECTOR_SIZE", 3)
    monkeypatch.setattr(cache, "get_qdrant_client", lambda: client)
    monkeypatch.setattr(cache, "embed_query", lambda text: [1.0, 0.0, 0.0])
    yield cache.SemanticCache()
    client.close()


def test_cache_keeps_sources_category_and_grade(semantic_cache):
    docs = [{"text": "Supply risk", "metadata": {"ticker": "AAPL", "cik": "0000320193"}}]
    semantic_cache.store("risks", "answer", retrieved_docs=docs, query_category="temporal")

    assert semantic_cache.lookup("risks") == {
        "final_response": "answer",
        "retrieved_docs": docs,
        "query_category": "temporal",
        "retrieval_grade": "relevant",
    }


def test_cache_isolates_company_filters_and_unfiltered_answers(semantic_cache):
    semantic_cache.store("risks", "Apple answer", company_filter=" aapl ")
    assert semantic_cache.lookup("risks", company_filter="MSFT") is None
    assert semantic_cache.lookup("risks") is None

    semantic_cache.store("risks", "Microsoft answer", company_filter="MSFT")
    semantic_cache.store("risks", "All companies answer")

    assert semantic_cache.lookup("risks", company_filter="AAPL")["final_response"] == "Apple answer"
    assert semantic_cache.lookup("risks", company_filter="msft")["final_response"] == "Microsoft answer"
    assert semantic_cache.lookup("risks")["final_response"] == "All companies answer"


def test_cache_ignores_legacy_answers_without_context(semantic_cache):
    semantic_cache.client.upsert(
        collection_name=semantic_cache.collection_name,
        points=[models.PointStruct(id=1, vector=[1.0, 0.0, 0.0], payload={"response_text": "legacy answer"})],
    )

    assert semantic_cache.lookup("risks") is None


def test_cache_respects_similarity_threshold(semantic_cache):
    semantic_cache.store("risks", "answer")

    with mock.patch.object(cache, "embed_query", return_value=[0.0, 1.0, 0.0]):
        assert semantic_cache.lookup("unrelated") is None
