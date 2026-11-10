"""Hybrid retrieval: dense and sparse search, fusion and reranking.

Responsibilities:
  - Run dense (Qdrant) and sparse (BM25) searches one after the other.
  - Merge and deduplicate the candidates with Reciprocal Rank Fusion.
  - Rerank candidates with a cross-encoder and drop low scores.
  - Detect tickers and regulations named in the query. A ticker sets
    the dense search filter. A regulation is only logged.

Dependencies:
  - The BGE reranker weights, downloaded on first use.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from sentence_transformers import CrossEncoder

from src.core import config
from src.core.constants import KNOWN_REGULATIONS, KNOWN_TICKERS
from src.core.lazy import locked_singleton
from src.core.logging import get_logger
from src.rag.bm25_index import sparse_search
from src.rag.vector_store import dense_search

logger = get_logger(__name__)
# Damping constant in the RRF score 1 / (k + rank).
_RRF_K = 60


@locked_singleton
def _get_reranker() -> CrossEncoder:
    """Return the shared BGE reranker, loading it on first call.

    Returns:
        The CrossEncoder named by config.BGE_RERANKER_MODEL_NAME.
    """
    return CrossEncoder(config.BGE_RERANKER_MODEL_NAME)


def _detect_regulation(query_text: str) -> str | None:
    """Return the first KNOWN_REGULATIONS entry in the query, or None.

    Matching is a case-insensitive substring test.

    Args:
        query_text: The raw user query string.
    """
    lowered = query_text.lower()
    for regulation in KNOWN_REGULATIONS:
        if regulation.lower() in lowered:
            return regulation
    return None


def _detect_ticker(query_text: str) -> str | None:
    """Return the first KNOWN_TICKERS entry in the query, or None.

    Uses word-boundary matching so that a ticker such as 'MS' does not
    match inside 'business' or 'most'.

    Args:
        query_text: The raw user query string.
    """
    for ticker in KNOWN_TICKERS:
        if re.search(rf"\b{re.escape(ticker)}\b", query_text, re.IGNORECASE):
            return ticker
    return None


def rrf_merge(result_lists: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Merge ranked result lists with Reciprocal Rank Fusion (RRF).

    Combines the outputs of several retrieval strategies without score
    normalization. Results with identical text are merged into one
    entry whose score sums the per-list RRF scores.

    Args:
        result_lists: The ranked result lists to fuse.

    Returns:
        The merged results, sorted by descending RRF score.
    """
    aggregated: dict[str, dict[str, Any]] = {}
    for result_list in result_lists:
        for rank, result in enumerate(result_list, start=1):
            key = hashlib.md5(str(result.get("text", "")).encode("utf-8")).hexdigest()
            rrf_score = 1.0 / (_RRF_K + rank)
            if key not in aggregated:
                aggregated[key] = {
                    "text": str(result.get("text", "")),
                    "metadata": dict(result.get("metadata", {})),
                    "score": rrf_score,
                }
            else:
                aggregated[key]["score"] += rrf_score
    merged = list(aggregated.values())
    merged.sort(key=lambda item: float(item["score"]), reverse=True)
    return merged


def _rerank(query_text: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Score and sort candidates with the BGE cross-encoder.

    Args:
        query_text: The user query string.
        candidates: Fused candidate documents from the RRF merge.

    Returns:
        The candidates sorted by descending cross-encoder score, or an
        empty list if there are none.
    """
    if not candidates:
        return []

    sentence_pairs = [[query_text, str(candidate.get("text", ""))] for candidate in candidates]
    scores = _get_reranker().predict(sentence_pairs)

    if isinstance(scores, float):
        scores = [scores]

    normalized: list[dict[str, Any]] = []
    for candidate, score in zip(candidates, scores):
        normalized.append(
            {
                "text": str(candidate.get("text", "")),
                "score": float(score),
                "metadata": dict(candidate.get("metadata", {})),
            }
        )

    normalized.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    return normalized


def retrieve(query_text: str, company_filter: str | None = None) -> list[dict[str, Any]]:
    """Run the hybrid retrieval pipeline and return the top results.

    Runs dense (Qdrant) and then sparse (BM25) retrieval, fuses the
    results with RRF, reranks with a cross-encoder, trims to the top k
    and drops results below config.BGE_SCORE_THRESHOLD.

    Args:
        query_text: The user query string.
        company_filter: Optional ticker symbol that restricts the dense
            search to one company. A ticker named in the query takes
            precedence. The sparse search is never filtered.

    Returns:
        Result dicts with 'text', 'score' and 'metadata' keys, whose
        score is at least config.BGE_SCORE_THRESHOLD.

    Raises:
        FileNotFoundError: If the BM25 index has not been built.

    Example:
        >>> docs = retrieve("Apple total net sales fiscal 2025")
        >>> docs[0]["metadata"]["ticker"]
        'AAPL'
    """
    regulation_match = _detect_regulation(query_text)
    if regulation_match:
        logger.info(
            "Regulation detected in query",
            extra={"extra": {"regulation": regulation_match}},
        )
    else:
        logger.info("No regulation detected in query")

    ticker_match = _detect_ticker(query_text)
    if ticker_match:
        logger.info("Ticker detected in query", extra={"extra": {"ticker": ticker_match}})
    else:
        logger.info("No ticker detected in query")

    effective_ticker = ticker_match or company_filter

    dense_results = dense_search(
        query_text,
        top_k=config.TOP_K_RETRIEVAL_COUNT,
        filter_ticker=effective_ticker if effective_ticker else None,
    )
    logger.info(
        "Dense retrieval complete",
        extra={"extra": {"count": len(dense_results)}},
    )

    sparse_results = sparse_search(query_text, top_k=config.TOP_K_RETRIEVAL_COUNT)
    logger.info(
        "Sparse retrieval complete",
        extra={"extra": {"count": len(sparse_results)}},
    )

    fused = rrf_merge([dense_results, sparse_results])
    logger.info(
        "RRF fusion complete",
        extra={"extra": {"count": len(fused)}},
    )

    reranked = _rerank(query_text, fused)
    logger.info(
        "BGE reranking complete",
        extra={"extra": {"count": len(reranked)}},
    )

    top_k_reranked = reranked[: config.TOP_K_RETRIEVAL_COUNT]
    logger.info(
        "Top-k trim complete",
        extra={"extra": {"count": len(top_k_reranked)}},
    )

    guarded = [item for item in top_k_reranked if float(item.get("score", 0.0)) >= config.BGE_SCORE_THRESHOLD]
    logger.info(
        "RetrievalGuard filtering complete",
        extra={"extra": {"count": len(guarded)}},
    )

    return guarded
