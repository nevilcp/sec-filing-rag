"""Unit tests for the hybrid retrieval pipeline.

Responsibilities:
  - Check that Reciprocal Rank Fusion merges and deduplicates results.
  - Check the ordering of cross-encoder reranking scores.
  - Check ticker and regulation detection in queries, including word
    boundary matching.
"""

from unittest import mock

from src.core import config
from src.core.constants import KNOWN_REGULATIONS, KNOWN_TICKERS
from src.rag.retriever import (
    _RRF_K,
    _detect_regulation,
    _detect_ticker,
    _rerank,
    retrieve,
    rrf_merge,
)


def test_detect_regulation():
    reg = KNOWN_REGULATIONS[0]
    assert _detect_regulation(f"What about {reg}?") == reg
    assert _detect_regulation("What about some random rule?") is None


def test_detect_ticker():
    ticker = KNOWN_TICKERS[0]
    assert _detect_ticker(f"How is {ticker} doing?") == ticker
    assert _detect_ticker("How is random company doing?") is None


def test_detect_ticker_no_false_positive():
    assert _detect_ticker("What are the latest earnings from that business?") is None
    assert _detect_ticker("What tobacco disclosures exist?") is None
    assert _detect_ticker("What are GS risk disclosures?") == "GS"
    result = _detect_ticker("Compare MS and JPM filings.")
    assert result in ("MS", "JPM")


def test_rrf_merge():
    list1 = [{"text": "docA", "metadata": {}}, {"text": "docB", "metadata": {}}]
    list2 = [{"text": "docC", "metadata": {}}, {"text": "docA", "metadata": {}}]

    merged = rrf_merge([list1, list2])

    assert len(merged) == 3
    assert merged[0]["text"] == "docA"
    assert merged[1]["text"] == "docC"
    assert merged[2]["text"] == "docB"
    assert merged[0]["score"] == (1.0 / (_RRF_K + 1)) + (1.0 / (_RRF_K + 2))


@mock.patch("src.rag.retriever._get_reranker")
def test_rerank(mock_get_reranker):
    mock_model = mock.Mock()
    mock_model.predict.return_value = [0.1, 0.9]
    mock_get_reranker.return_value = mock_model

    candidates = [{"text": "bad match"}, {"text": "good match"}]
    reranked = _rerank("query", candidates)

    assert len(reranked) == 2
    assert reranked[0]["text"] == "good match"
    assert reranked[0]["score"] == 0.9
    assert reranked[1]["text"] == "bad match"
    assert reranked[1]["score"] == 0.1


@mock.patch("src.rag.retriever.dense_search")
@mock.patch("src.rag.retriever.sparse_search")
@mock.patch("src.rag.retriever._rerank")
def test_retrieve_pipeline(mock_rerank, mock_sparse, mock_dense):
    mock_dense.return_value = [{"text": "dense doc", "metadata": {}}]
    mock_sparse.return_value = [{"text": "sparse doc", "metadata": {}}]

    mock_rerank.return_value = [
        {"text": "good doc", "score": 10.0, "metadata": {}},
        {"text": "bad doc", "score": -5.0, "metadata": {}},
    ]

    config.BGE_SCORE_THRESHOLD = 0.0

    results = retrieve("query without ticker")

    assert len(results) == 1
    assert results[0]["text"] == "good doc"
    assert results[0]["score"] == 10.0


@mock.patch("src.rag.retriever.dense_search")
@mock.patch("src.rag.retriever.sparse_search")
@mock.patch("src.rag.retriever._rerank")
def test_retrieve_fallback_when_all_filtered(mock_rerank, mock_sparse, mock_dense):
    mock_dense.return_value = []
    mock_sparse.return_value = []

    mock_rerank.return_value = [
        {"text": "bad doc 1", "score": -5.0, "metadata": {}},
        {"text": "bad doc 2", "score": -6.0, "metadata": {}},
    ]

    config.BGE_SCORE_THRESHOLD = 0.0
    results = retrieve("query")

    assert len(results) == 0
