"""
tests/unit/test_nodes.py

Unit tests for the LangGraph state machine nodes.

Responsibilities:
  - Verify intent categorization and routing
  - Test dual-gate (math + semantic) retrieval grading
  - Ensure the CRAG rewriting loop terminates correctly
  - Validate context formatting for final synthesis
"""

from unittest import mock

from src.agent.nodes import (
    cache_check,
    grade_retrieval,
    process_input,
    respond,
    retrieve_and_reason,
    rewrite_query,
)
from src.agent.state import ComplianceState
from src.core import config


@mock.patch("src.agent.nodes.invoke_with_rate_limit")
def test_process_input_valid_json(mock_invoke):
    mock_invoke.return_value = '{"query_category": "temporal"}'
    state = ComplianceState(query="test")
    res = process_input(state)
    assert res["query_category"] == "temporal"


@mock.patch("src.agent.nodes.invoke_with_rate_limit")
def test_process_input_fallback_string(mock_invoke):
    mock_invoke.return_value = "this looks like a cross_company question"
    state = ComplianceState(query="test")
    res = process_input(state)
    assert res["query_category"] == "cross_company"


@mock.patch("src.agent.nodes._cache")
def test_cache_check_hit(mock_cache):
    mock_cache.return_value.lookup.return_value = "cached answer"
    state = ComplianceState(query="test")
    res = cache_check(state)
    assert res["cache_hit"] is True
    assert res["final_response"] == "cached answer"


@mock.patch("src.agent.nodes._cache")
def test_cache_check_miss(mock_cache):
    mock_cache.return_value.lookup.return_value = None
    state = ComplianceState(query="test")
    res = cache_check(state)
    assert res["cache_hit"] is False


@mock.patch("src.agent.nodes.retrieve")
@mock.patch("src.agent.nodes.invoke_with_rate_limit")
def test_retrieve_and_reason_relevant(mock_invoke, mock_retrieve):
    mock_retrieve.return_value = [{"score": 1.0, "text": "Good doc"}]
    config.RELEVANCE_MEAN_THRESHOLD = 0.5
    config.RELEVANCE_TOP1_THRESHOLD = 0.5

    mock_invoke.return_value = "relevant"
    state = ComplianceState(query="test")

    res = retrieve_and_reason(state)
    assert res["context_relevance"] == "relevant"
    assert len(res["retrieved_docs"]) == 1


@mock.patch("src.agent.nodes.retrieve")
@mock.patch("src.agent.nodes.invoke_with_rate_limit")
def test_retrieve_and_reason_irrelevant_math(mock_invoke, mock_retrieve):
    mock_retrieve.return_value = [{"score": 0.1, "text": "Bad doc"}]
    config.RELEVANCE_MEAN_THRESHOLD = 0.5
    config.RELEVANCE_TOP1_THRESHOLD = 0.5

    mock_invoke.return_value = "relevant"
    state = ComplianceState(query="test")

    res = retrieve_and_reason(state)
    assert res["context_relevance"] == "irrelevant"


def test_grade_retrieval():
    state = ComplianceState(query="test", context_relevance="relevant")
    res = grade_retrieval(state)
    assert res["retrieval_grade"] == "relevant"


@mock.patch("src.agent.nodes.invoke_with_rate_limit")
def test_rewrite_query_under_limit(mock_invoke):
    mock_invoke.return_value = "rewritten text"
    config.MAX_CRAG_REWRITES = 3
    state = ComplianceState(query="test", rewrite_count=0)

    res = rewrite_query(state)
    assert res["query"] == "rewritten text"
    assert res["rewrite_count"] == 1
    assert res["context_relevance"] is None


def test_rewrite_query_over_limit():
    config.MAX_CRAG_REWRITES = 3
    state = ComplianceState(query="test", rewrite_count=3)

    res = rewrite_query(state)
    assert res["retrieval_grade"] == "relevant"


@mock.patch("src.agent.nodes.invoke_with_rate_limit")
@mock.patch("src.agent.nodes._cache")
def test_respond_normal(mock_cache, mock_invoke):
    mock_invoke.return_value = "Generated response"
    state = ComplianceState(
        query="test",
        retrieved_docs=[
            {
                "text": "doc1",
                "metadata": {
                    "ticker": "AAPL",
                    "form_type": "10-K",
                    "filing_date": "2024-01-01",
                    "section_name": "mda",
                },
            }
        ],
        cache_hit=False,
    )

    res = respond(state)
    assert "Generated response" in res["final_response"]
    assert "Sources:" in res["final_response"]
    assert "AAPL | 10-K | 2024-01-01 | mda" in res["final_response"]
    mock_cache.return_value.store.assert_called_once_with("test", "Generated response")


def test_respond_cache_hit():
    state = ComplianceState(query="test", cache_hit=True, final_response="This was from cache.")
    res = respond(state)
    assert "This was from cache." in res["final_response"]


@mock.patch("src.agent.nodes.invoke_with_rate_limit")
@mock.patch("src.agent.nodes._cache")
def test_respond_irrelevant_grade_short_circuits_without_caching(mock_cache, mock_invoke):
    state = ComplianceState(query="test", retrieval_grade="irrelevant", cache_hit=False)

    res = respond(state)

    assert "could not find sufficient information" in res["final_response"]
    mock_invoke.assert_not_called()
    mock_cache.return_value.store.assert_not_called()
