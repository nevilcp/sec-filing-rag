"""
rag/bm25_index.py

BM25 sparse index build/load/search utilities.

Responsibilities:
  - Build and persist a local BM25 keyword index from the Qdrant document corpus
  - Load the BM25 index and its associated metadata into memory
  - Execute sparse retrieval for exact-match terminology

Dependencies:
  - bm25s
"""

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
    """Lowercase and tokenize text into word-boundary-delimited tokens.

    Uses a simple word-boundary regex rather than an NLP tokenizer to
    keep the BM25 index lightweight and deterministic across platforms.

    Args:
        text: Raw input text to tokenize.

    Returns:
        A list of lowercase word tokens.
    """
    return TOKEN_PATTERN.findall(text.lower())


def build_bm25_index(documents: list[dict[str, Any]]) -> None:
    """Build and persist a BM25 keyword index from the provided documents.

    Tokenizes each document's 'text' field, fits a BM25 model, and saves
    both the index and the original document list to the configured path.
    The document list is saved separately so sparse_search can reconstruct
    full result dicts without re-querying Qdrant.

    Args:
        documents: A list of document dicts, each containing a 'text'
            key and a 'metadata' key with filing provenance fields.
    """
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
    """Load the persisted BM25 index and document list from disk.

    Returns:
        A dict with keys 'index' (a bm25s.BM25 instance) and 'documents'
        (the list of document dicts saved during ingestion).

    Raises:
        FileNotFoundError: If the BM25 index path does not exist.
    """
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
    """Return the cached BM25 payload, loading from disk on first call.

    The payload is retained for the lifetime of the process to avoid
    repeated disk I/O on every sparse_search call.

    Returns:
        A dict with 'index' and 'documents' keys as returned by
        load_bm25_index.
    """
    global _BM25_PAYLOAD
    with _BM25_LOCK:
        if _BM25_PAYLOAD is None:
            _BM25_PAYLOAD = load_bm25_index()
        return _BM25_PAYLOAD


def sparse_search(query_text: str, top_k: int) -> list[dict[str, Any]]:
    """Score all indexed documents with BM25 and return the top-k results.

    Args:
        query_text: The user query string to tokenize and score against.
        top_k: Maximum number of results to return.

    Returns:
        A list of result dicts with 'text', 'score', and 'metadata' keys,
        sorted by descending BM25 score. Returns an empty list if the
        query tokenizes to nothing or the index has no documents.
    """
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
