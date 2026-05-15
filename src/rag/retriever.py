"""Hybrid retrieval pipeline: dense + sparse + regulation-directed + reranking."""

from __future__ import annotations

import hashlib
from typing import Any

from sentence_transformers import CrossEncoder

from src.core import config
from src.core.constants import KNOWN_REGULATIONS, KNOWN_TICKERS
from src.core.logging import get_logger
from src.rag.bm25_index import sparse_search
from src.rag.vector_store import dense_search

logger = get_logger(__name__)
_BGE_RERANKER: CrossEncoder | None = None
_RRF_K = 60


def _get_reranker() -> CrossEncoder:
    global _BGE_RERANKER
    if _BGE_RERANKER is None:
        _BGE_RERANKER = CrossEncoder("BAAI/bge-reranker-v2-m3")
    return _BGE_RERANKER


def _detect_regulation(query_text: str) -> str | None:
    lowered = query_text.lower()
    for regulation in KNOWN_REGULATIONS:
        if regulation.lower() in lowered:
            return regulation
    return None


def _detect_ticker(query_text: str) -> str | None:
    lowered = query_text.lower()
    for ticker in KNOWN_TICKERS:
        if ticker.lower() in lowered:
            return ticker
    return None


def rrf_merge(result_lists: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Merge ranked results with Reciprocal Rank Fusion (RRF)."""
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


def _deduplicate_by_text_md5(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for item in results:
        text = str(item.get("text", ""))
        key = hashlib.md5(text.encode("utf-8")).hexdigest()
        current_score = float(item.get("score", 0.0))
        existing = deduped.get(key)
        if existing is None or current_score > float(existing.get("score", 0.0)):
            deduped[key] = item
    output = list(deduped.values())
    output.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    return output


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


def retrieve(query_text: str) -> list[dict[str, Any]]:
    """Run full hybrid retrieval and return final top-k results."""
    regulation_match = _detect_regulation(query_text)
    if regulation_match:
        logger.info("Regulation detected in query", extra={"extra": {"regulation": regulation_match}})
    else:
        logger.info("No regulation detected in query")

    ticker_match = _detect_ticker(query_text)
    if ticker_match:
        logger.info("Ticker detected in query", extra={"extra": {"ticker": ticker_match}})
    else:
        logger.info("No ticker detected in query")

    dense_results = dense_search(query_text, top_k=config.TOP_K_RETRIEVAL_COUNT)
    logger.info(
        "Dense retrieval complete",
        extra={"extra": {"count": len(dense_results)}},
    )

    sparse_results = sparse_search(query_text, top_k=config.TOP_K_RETRIEVAL_COUNT)
    logger.info(
        "Sparse retrieval complete",
        extra={"extra": {"count": len(sparse_results)}},
    )

    directed_results: list[dict[str, Any]] = []
    if regulation_match or ticker_match:
        kwargs: dict[str, Any] = {"top_k": 5}
        if ticker_match:
            kwargs["filter_ticker"] = ticker_match
        directed_results = dense_search(query_text, **kwargs)
        
    logger.info(
        "Directed retrieval complete",
        extra={"extra": {"count": len(directed_results)}},
    )

    fused = rrf_merge([dense_results, sparse_results, directed_results])
    logger.info(
        "RRF fusion complete",
        extra={"extra": {"count": len(fused)}},
    )

    deduped = _deduplicate_by_text_md5(fused)
    logger.info(
        "Deduplication complete",
        extra={"extra": {"count": len(deduped)}},
    )

    reranked = _rerank(query_text, deduped)
    logger.info(
        "BGE reranking complete",
        extra={"extra": {"count": len(reranked)}},
    )

    top_k_reranked = reranked[: config.TOP_K_RETRIEVAL_COUNT]
    logger.info(
        "Top-k trim complete",
        extra={"extra": {"count": len(top_k_reranked)}},
    )

    guarded = [
        item
        for item in top_k_reranked
        if float(item.get("score", 0.0)) >= config.BGE_SCORE_THRESHOLD
    ]
    logger.info(
        "RetrievalGuard filtering complete",
        extra={"extra": {"count": len(guarded)}},
    )

    if not guarded and top_k_reranked:
        logger.warning(
            "All chunks filtered by RetrievalGuard; returning full reranked list."
        )
        return top_k_reranked
    return guarded

