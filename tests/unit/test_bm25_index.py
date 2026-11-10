"""
Unit tests for the BM25 index build and search cache.
"""

from __future__ import annotations

from src.core import config
from src.rag import bm25_index


def _doc(text: str) -> dict:
    return {"text": text, "metadata": {"ticker": "AAPL"}}


def test_sparse_search_after_rebuild_returns_new_documents(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BM25_INDEX_PATH", str(tmp_path / "bm25"))
    monkeypatch.setattr(bm25_index, "_BM25_PAYLOAD", None)

    bm25_index.build_bm25_index([_doc("alpha revenue"), _doc("beta margin")])
    assert bm25_index.sparse_search("alpha", top_k=1)[0]["text"] == "alpha revenue"

    bm25_index.build_bm25_index([_doc("gamma 394,328 sales"), _doc("delta risk")])

    top = bm25_index.sparse_search("gamma", top_k=1)
    assert top[0]["text"] == "gamma 394,328 sales"
