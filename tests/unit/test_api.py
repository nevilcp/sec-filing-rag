"""HTTP and graph integration checks with local storage and deterministic model boundaries."""

import json
from unittest import mock

import pytest
from fastapi.testclient import TestClient

from src.agent import nodes
from src.api import main, routes
from src.core import config
from src.rag import cache, vector_store


@pytest.fixture
def api_app(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "QDRANT_PATH", str(tmp_path / "qdrant"))
    monkeypatch.setattr(config, "QDRANT_VECTOR_SIZE", 3)
    monkeypatch.setattr(config, "INGEST_CHECKPOINT_PATH", str(tmp_path / "checkpoint.json"))
    monkeypatch.setattr(config, "RELEVANCE_MEAN_THRESHOLD", 0.5)
    monkeypatch.setattr(config, "RELEVANCE_TOP1_THRESHOLD", 0.7)
    monkeypatch.setattr(config, "MAX_CRAG_REWRITES", 2)
    monkeypatch.setattr(cache, "embed_query", lambda text: [1.0, 0.0, 0.0])
    vector_store.get_filings_collection.cache_clear()
    vector_store.get_qdrant_client.cache_clear()
    nodes._cache.cache_clear()
    monkeypatch.setattr(routes, "validate_input", lambda text: (True, ""))
    monkeypatch.setattr(routes, "validate_output", lambda text: (text, []))
    yield main.app
    nodes._cache.cache_clear()


def test_query_cache_preserves_citations_and_metrics_across_sessions(api_app, monkeypatch):
    doc = {
        "text": "Supply chain risks",
        "score": 0.99,
        "metadata": {"ticker": "AAPL", "cik": "0000320193", "accession_number": "0000320193-25-000079"},
    }
    retrieve = mock.Mock(return_value=[doc])
    invoke = mock.Mock(side_effect=['{"query_category":"temporal"}', "relevant", "Informational answer"])
    monkeypatch.setattr(nodes, "retrieve", retrieve)
    monkeypatch.setattr(nodes, "invoke_with_rate_limit", invoke)

    with TestClient(api_app) as client:
        payload = {"query": "Supply chain risks?", "company_filter": "AAPL", "session_id": "first"}
        first = client.post("/query", json=payload)
        second = client.post("/query", json={**payload, "session_id": "second"})
        assert first.status_code == second.status_code == 200
        fresh, cached = first.json(), second.json()
        assert fresh["cache_hit"] is False
        assert cached["cache_hit"] is True
        assert cached["sources"] == fresh["sources"]
        assert cached["sources"][0]["cik"] == "0000320193"
        assert cached["query_category"] == "temporal"
        assert cached["retrieval_grade"] == "relevant"
        assert cached["retrieved_doc_count"] == 1
        retrieve.assert_called_once()
        assert invoke.call_count == 3
        metrics = client.get("/metrics").json()
        assert metrics["total_queries_served"] == 2
        assert metrics["cache_hit_rate"] == 0.5

    # Reopening the app must acquire a new client and reuse the persistent cache safely.
    with TestClient(api_app) as client:
        cached = client.post("/query", json={**payload, "session_id": "third"})
        assert cached.status_code == 200
        assert cached.json()["sources"] == fresh["sources"]
        assert cached.json()["cache_hit"] is True
        assert invoke.call_count == 3


def test_irrelevant_retrieval_stops_after_rewrite_limit(api_app, monkeypatch):
    retrieve = mock.Mock(return_value=[])
    invoke = mock.Mock(side_effect=['{"query_category":"general"}', "rewritten once", "rewritten twice"])
    monkeypatch.setattr(nodes, "retrieve", retrieve)
    monkeypatch.setattr(nodes, "invoke_with_rate_limit", invoke)

    with TestClient(api_app) as client:
        response = client.post("/query", json={"query": "missing information", "session_id": "retry"})
        assert response.status_code == 200
        data = response.json()
        assert data["retrieval_grade"] == "irrelevant"
        assert data["rewrite_count"] == 2
        assert data["sources"] == []
        assert "could not find sufficient information" in data["response"]
        assert retrieve.call_count == 3


def test_api_validation_status_and_ingestion_trigger(api_app, monkeypatch):
    ingest = mock.Mock(return_value=0)
    monkeypatch.setattr(routes, "run_ingestion", ingest)
    with TestClient(api_app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/corpus/status").status_code == 503
        with open(config.INGEST_CHECKPOINT_PATH, "w") as f:
            json.dump({"indexed_companies": ["AAPL"]}, f)
        assert client.get("/corpus/status").json()["indexed_companies"] == ["AAPL"]
        assert client.post("/query", json={"query": "", "session_id": "empty"}).status_code == 422
        monkeypatch.setattr(routes, "validate_input", lambda text: (False, "PII detected"))
        assert client.post("/query", json={"query": "PII", "session_id": "blocked"}).status_code == 400
        assert client.get("/metrics").json()["total_queries_served"] == 0
        assert client.post("/ingest/trigger").json() == {"status": "ingestion_started"}
        ingest.assert_called_once()
