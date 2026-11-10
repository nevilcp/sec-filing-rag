"""Unit tests for the LangGraph nodes.

Responsibilities:
  - Check query categorization, including unparseable model replies.
  - Check retrieval grading, which needs both the score check and the
    LLM verdict to pass.
  - Check the rewrite limit, the semantic cache lookup and the final
    response, including its Sources footer.
"""

from unittest import mock

import pytest

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
    stored_query, stored_text = mock_cache.return_value.store.call_args.args
    assert stored_query == "test"
    assert stored_text == res["final_response"]


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


@pytest.mark.parametrize("reply", ['"general"', "[]", '"temporal"'])
@mock.patch("src.agent.nodes.invoke_with_rate_limit")
def test_process_input_non_object_json_returns_category_without_crashing(mock_invoke, reply):
    mock_invoke.return_value = reply

    res = process_input(ComplianceState(query="test"))

    assert res["query_category"] == ("temporal" if "temporal" in reply else "general")


@pytest.mark.parametrize(
    ("verdict", "expected"),
    [
        ("relevant", "relevant"),
        ("Relevant", "relevant"),
        ("relevant.", "relevant"),
        ("not relevant", "irrelevant"),
        ("irrelevant.", "irrelevant"),
        ("Irrelevant", "irrelevant"),
        ("", "irrelevant"),
    ],
)
@mock.patch("src.agent.nodes.retrieve")
@mock.patch("src.agent.nodes.invoke_with_rate_limit")
def test_retrieve_and_reason_grader_verdict_decides_relevance(mock_invoke, mock_retrieve, verdict, expected):
    mock_retrieve.return_value = [{"score": 1.0, "text": "doc"}]
    mock_invoke.return_value = verdict

    res = retrieve_and_reason(ComplianceState(query="test"))

    assert res["context_relevance"] == expected


@mock.patch("src.agent.nodes._cache")
def test_cache_check_uses_original_query_after_rewrite(mock_cache):
    mock_cache.return_value.lookup.return_value = None

    cache_check(ComplianceState(query="rewritten", original_query="original"))

    mock_cache.return_value.lookup.assert_called_once_with("original")


@mock.patch("src.agent.nodes.invoke_with_rate_limit")
@mock.patch("src.agent.nodes._cache")
def test_respond_after_rewrite_stores_final_text_under_original_query(mock_cache, mock_invoke):
    mock_invoke.return_value = "Generated response"
    doc = {"text": "t", "metadata": {"ticker": "AAPL", "form_type": "10-K", "filing_date": "2024-01-01"}}
    state = ComplianceState(query="rewritten", original_query="original", retrieved_docs=[doc], cache_hit=False)

    res = respond(state)

    stored_query, stored_text = mock_cache.return_value.store.call_args.args
    assert stored_query == "original"
    assert "Sources:" in stored_text
    assert stored_text == res["final_response"]
    assert "<user_query>\noriginal\n</user_query>" in mock_invoke.call_args.args[1]


def test_respond_cache_hit_returns_stored_text_with_sources_unchanged():
    stored = "Answer.\n\nSources:\n- AAPL | 10-K | 2024-01-01 | mda (Accession: 1)"
    doc = {"text": "t", "metadata": {"ticker": "MSFT"}}

    res = respond(ComplianceState(query="q", cache_hit=True, final_response=stored, retrieved_docs=[doc]))

    assert res["final_response"] == stored
