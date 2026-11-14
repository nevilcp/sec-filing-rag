"""Hybrid retrieval: dense and sparse search, fusion and reranking."""

from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING, Any

from src.core import config
from src.core.constants import KNOWN_REGULATIONS, KNOWN_TICKERS
from src.core.lazy import locked_singleton
from src.core.logging import get_logger
from src.rag.bm25_index import sparse_search
from src.rag.vector_store import dense_search

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

logger = get_logger(__name__)
# Damping constant in the RRF score 1 / (k + rank).
_RRF_K = 60


@locked_singleton
def _get_reranker() -> CrossEncoder:
    from sentence_transformers import CrossEncoder

    return CrossEncoder(config.BGE_RERANKER_MODEL_NAME)


def _detect_regulation(query_text: str) -> str | None:
    """Return the first configured regulation found by case-insensitive substring matching."""
    lowered = query_text.lower()
    for regulation in KNOWN_REGULATIONS:
        if regulation.lower() in lowered:
            return regulation
    return None


def _detect_ticker(query_text: str) -> str | None:
    """Return the first configured ticker found as a whole word, ignoring case."""
    for ticker in KNOWN_TICKERS:
        if re.search(rf"\b{re.escape(ticker)}\b", query_text, re.IGNORECASE):
            return ticker
    return None


def rrf_merge(result_lists: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Fuse ranked lists with RRF, merging identical text and retaining the first metadata."""
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
    """Fuse dense and sparse matches, rerank them, trim to top k, and drop low scores."""
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

    effective_ticker = ticker_match or (company_filter.strip().upper() if company_filter else None)

    dense_results = dense_search(
        query_text,
        top_k=config.TOP_K_RETRIEVAL_COUNT,
        filter_ticker=effective_ticker if effective_ticker else None,
    )
    logger.info(
        "Dense retrieval complete",
        extra={"extra": {"count": len(dense_results)}},
    )

    sparse_results = sparse_search(query_text, top_k=config.TOP_K_RETRIEVAL_COUNT, filter_ticker=effective_ticker)
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
