"""BM25 sparse index build, load and search utilities."""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any

import bm25s

from src.core import config

TOKEN_PATTERN = re.compile(r"\b\w+\b")


def _tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def build_bm25_index(documents: list[dict[str, Any]]) -> None:
    """Persist the BM25 index and its documents, then invalidate the in-memory payload."""
    corpus_tokens = [_tokenize(str(doc.get("text", ""))) for doc in documents]

    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)

    path = Path(config.BM25_INDEX_PATH)
    path.mkdir(parents=True, exist_ok=True)

    retriever.save(str(path))

    docs_path = path / "documents.json"
    with docs_path.open("w", encoding="utf-8") as f:
        json.dump(documents, f)

    # Drop the cached payload so the next search reloads the new index.
    global _BM25_PAYLOAD
    with _BM25_LOCK:
        _BM25_PAYLOAD = None


def load_bm25_index() -> dict[str, Any]:
    """Load the saved BM25 index and documents, raising if the index directory is missing."""
    path = Path(config.BM25_INDEX_PATH)
    if not path.exists():
        raise FileNotFoundError(
            "BM25 index not found at configured path. Run ingestion first to build retrieval artifacts."
        )

    retriever = bm25s.BM25.load(str(path), load_corpus=False)

    docs_path = path / "documents.json"
    with docs_path.open("r", encoding="utf-8") as f:
        documents = json.load(f)

    return {
        "index": retriever,
        "documents": documents,
    }


_BM25_PAYLOAD: dict[str, Any] | None = None
_BM25_LOCK = threading.Lock()


def _get_bm25_payload() -> dict[str, Any]:
    global _BM25_PAYLOAD
    with _BM25_LOCK:
        if _BM25_PAYLOAD is None:
            _BM25_PAYLOAD = load_bm25_index()
        return _BM25_PAYLOAD


def sparse_search(query_text: str, top_k: int) -> list[dict[str, Any]]:
    """Return BM25 matches, or an empty list for a query without tokens or an empty corpus."""
    tokenized_query = _tokenize(query_text)

    if not tokenized_query:
        return []

    payload = _get_bm25_payload()
    index: bm25s.BM25 = payload["index"]
    documents: list[dict[str, Any]] = payload["documents"]

    if not documents:
        return []

    results, scores = index.retrieve([tokenized_query], corpus=documents, k=top_k)

    doc_results = results[0]
    doc_scores = scores[0]

    normalized_results: list[dict[str, Any]] = []
    for doc, score in zip(doc_results, doc_scores):
        normalized_results.append(
            {
                "text": str(doc.get("text", "")),
                "score": float(score),
                "metadata": dict(doc.get("metadata", {})),
            }
        )
    return normalized_results
