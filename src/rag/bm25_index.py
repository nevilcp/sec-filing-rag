"""BM25 sparse index build, load and search utilities.

Responsibilities:
  - Build and persist a local BM25 keyword index of the document corpus.
  - Load the index and its document list into memory.
  - Run sparse retrieval for exact-match terminology.
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
    """Lowercase text and split it into word tokens.

    Uses a word-boundary regex rather than an NLP tokenizer, which keeps
    the index light and deterministic across platforms.

    Args:
        text: Raw input text to tokenize.

    Returns:
        The lowercase word tokens.
    """
    return TOKEN_PATTERN.findall(text.lower())


def build_bm25_index(documents: list[dict[str, Any]]) -> None:
    """Build and persist a BM25 keyword index from the documents.

    Tokenizes each document's 'text' field, fits a BM25 model and saves
    the index and the document list to config.BM25_INDEX_PATH. The list
    is stored beside the index so sparse_search can return full result
    dicts without querying Qdrant. The in-memory cache is cleared so
    the next search loads the new index.

    Args:
        documents: Document dicts, each with a 'text' key and a
            'metadata' key holding filing provenance fields.

    Raises:
        OSError: If the index files cannot be written.
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
        A dict with 'index' (a bm25s.BM25 instance) and 'documents'
        (the document dicts saved during ingestion).

    Raises:
        FileNotFoundError: If the BM25 index path does not exist.
        json.JSONDecodeError: If the saved document list is invalid.
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
    """Return the cached BM25 payload, loading it from disk if needed.

    The payload is kept until build_bm25_index clears it, which avoids
    disk reads on every search.

    Returns:
        The dict returned by load_bm25_index.

    Raises:
        FileNotFoundError: If the BM25 index has not been built.
    """
    global _BM25_PAYLOAD
    with _BM25_LOCK:
        if _BM25_PAYLOAD is None:
            _BM25_PAYLOAD = load_bm25_index()
        return _BM25_PAYLOAD


def sparse_search(query_text: str, top_k: int) -> list[dict[str, Any]]:
    """Score all indexed documents with BM25 and return the top results.

    Args:
        query_text: The user query string to tokenize and score.
        top_k: Maximum number of results to return.

    Returns:
        Result dicts with 'text', 'score' and 'metadata' keys, sorted
        by descending BM25 score. The list is empty if the query has no
        tokens or the index has no documents.

    Raises:
        FileNotFoundError: If the BM25 index has not been built.
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
