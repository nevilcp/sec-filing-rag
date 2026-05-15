"""BM25 sparse index build/load/search utilities."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import bm25s

from src.core import config

TOKEN_PATTERN = re.compile(r"\b\w+\b")


def _tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def build_bm25_index(documents: list[dict[str, Any]]) -> None:
    """Build and persist BM25 index and its source documents."""
    corpus_tokens = [_tokenize(str(doc.get("text", ""))) for doc in documents]
    
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)

    path = Path(config.BM25_INDEX_PATH)
    path.mkdir(parents=True, exist_ok=True)
    
    retriever.save(str(path))
    
    docs_path = path / "documents.json"
    with docs_path.open("w", encoding="utf-8") as f:
        json.dump(documents, f)


def load_bm25_index() -> dict[str, Any]:
    """Load persisted BM25 index payload from disk."""
    path = Path(config.BM25_INDEX_PATH)
    if not path.exists():
        raise FileNotFoundError(
            "BM25 index not found at configured path. "
            "Run ingestion first to build retrieval artifacts."
        )
    
    retriever = bm25s.BM25.load(str(path), load_corpus=False)
    
    docs_path = path / "documents.json"
    with docs_path.open("r", encoding="utf-8") as f:
        documents = json.load(f)
        
    return {
        "index": retriever,
        "documents": documents,
    }


def sparse_search(query_text: str, top_k: int) -> list[dict[str, Any]]:
    """Score all indexed docs with BM25 and return top-k normalized results."""
    payload = load_bm25_index()
    index: bm25s.BM25 = payload["index"]
    documents: list[dict[str, Any]] = payload["documents"]

    tokenized_query = _tokenize(query_text)
    
    if not documents:
        return []

    # bm25s.retrieve normally expects a list of queries. If we pass [tokenized_query],
    # it returns 2D arrays/lists. 
    # If we pass tokenized_query directly, it might interpret it as multiple queries if 
    # it's a list of strings. To be safe we wrap it, but we follow the shape required.
    results, scores = index.retrieve([tokenized_query], corpus=documents, k=top_k)
    
    # Extract the first query's results
    doc_results = results[0]
    doc_scores = scores[0]

    normalized_results: list[dict[str, Any]] = []
    # bm25s can return docs and scores as numpy arrays or iterables
    for doc, score in zip(doc_results, doc_scores):
        normalized_results.append(
            {
                "text": str(doc.get("text", "")),
                "score": float(score),
                "metadata": dict(doc.get("metadata", {})),
            }
        )
    return normalized_results
