"""
tests/unit/test_ingest.py

Unit tests for the EDGAR ingestion pipeline helpers.

Responsibilities:
  - Validate checkpoint load/save round-trip correctness
  - Verify stale-date detection resets the checkpoint date
  - Test the sub-batch splitter boundaries
  - Ensure embedding count mismatches skip the affected batch
"""

from __future__ import annotations

import json
from unittest import mock

import pytest

from src.core import config
from src.rag import ingest
from src.rag.ingest import _iter_sub_batches, _load_checkpoint, _save_checkpoint


def test_load_checkpoint_missing_file(tmp_path):
    """Returns a fresh default dict when no checkpoint file exists"""
    cp = _load_checkpoint(tmp_path / "missing.json")
    assert cp["processed_files"] == []
    assert cp["indexed_companies"] == []
    assert isinstance(cp["date"], str)


def test_load_checkpoint_empty_file(tmp_path):
    """Handles an empty checkpoint file gracefully"""
    path = tmp_path / "checkpoint.json"
    path.write_text("", encoding="utf-8")
    cp = _load_checkpoint(path)
    assert cp["processed_files"] == []


def test_save_and_load_checkpoint_roundtrip(tmp_path):
    """Saved checkpoint survives a load/save round-trip intact"""
    path = tmp_path / "checkpoint.json"
    data = {
        "date": "2024-01-01",
        "processed_files": ["0001234567-24-000001"],
        "indexed_companies": ["AAPL"],
    }
    _save_checkpoint(path, data)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["processed_files"] == ["0001234567-24-000001"]
    assert loaded["indexed_companies"] == ["AAPL"]


def test_load_checkpoint_stale_date_is_reset(tmp_path):
    """A checkpoint from a past date has its date field replaced with today"""
    path = tmp_path / "checkpoint.json"
    stale = {
        "date": "2020-01-01",
        "processed_files": ["old-file"],
        "indexed_companies": ["MSFT"],
    }
    path.write_text(json.dumps(stale), encoding="utf-8")
    cp = _load_checkpoint(path)
    assert cp["date"] != "2020-01-01"
    assert "old-file" in cp["processed_files"]


def test_iter_sub_batches_exact_split():
    items = list(range(10))
    batches = _iter_sub_batches(items, 5)
    assert len(batches) == 2
    assert batches[0] == [0, 1, 2, 3, 4]
    assert batches[1] == [5, 6, 7, 8, 9]


def test_iter_sub_batches_partial_last():
    items = list(range(7))
    batches = _iter_sub_batches(items, 5)
    assert len(batches) == 2
    assert batches[1] == [5, 6]


def test_iter_sub_batches_empty():
    assert _iter_sub_batches([], 5) == []


def test_iter_sub_batches_single_item():
    assert _iter_sub_batches([42], 5) == [[42]]


def test_save_checkpoint_interrupted_write_keeps_previous_file(tmp_path):
    path = tmp_path / "checkpoint.json"
    _save_checkpoint(path, {"date": "2024-01-01", "processed_files": ["a"], "indexed_companies": []})

    with mock.patch("src.core.fileio.os.replace", side_effect=KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt):
            _save_checkpoint(path, {"date": "2024-01-01", "processed_files": ["a", "b"], "indexed_companies": []})

    assert _load_checkpoint(path)["processed_files"] == ["a"]
    assert list(tmp_path.iterdir()) == [path]


def _run_ingestion_with_two_batches(tmp_path, monkeypatch, embed_side_effect):
    """Run ingestion for one filing that splits into two embedding batches."""
    checkpoint_path = tmp_path / "checkpoint.json"
    html_path = tmp_path / "filing.html"
    html_path.write_text("<html></html>", encoding="utf-8")
    monkeypatch.setattr(config, "INGEST_CHECKPOINT_PATH", str(checkpoint_path))
    monkeypatch.setattr(config, "EMBEDDING_BATCH_SIZE", 1)
    sections = {"business_overview": "First paragraph.\n\n" + "x" * config.CHUNK_SIZE}
    filing = {"accession_number": "0000000000-24-000001", "primary_document": "a.htm", "form_type": "10-K"}

    monkeypatch.setattr(ingest, "get_corpus_companies", lambda: [{"ticker": "AAPL", "company_name": "Apple"}])
    monkeypatch.setattr(ingest, "resolve_cik", lambda ticker: "0000320193")
    monkeypatch.setattr(ingest, "embed_documents", embed_side_effect)
    monkeypatch.setattr(ingest, "upsert_chunks", mock.Mock())
    monkeypatch.setattr(ingest, "get_qdrant_client", lambda: mock.Mock(scroll=lambda **kw: ([], None)))
    monkeypatch.setattr(ingest, "get_filings_collection", lambda: None)
    monkeypatch.setattr(ingest.edgar_client, "get_recent_filings", lambda *a, **kw: [filing])
    monkeypatch.setattr(ingest.edgar_client, "stage_filing", lambda **kw: html_path)
    monkeypatch.setattr(ingest.section_extractor, "extract_sections", lambda html: sections)

    ingest.run_ingestion()
    return json.loads(checkpoint_path.read_text(encoding="utf-8"))


def test_run_ingestion_failed_embedding_batch_leaves_filing_unprocessed(tmp_path, monkeypatch):
    calls = {"n": 0}

    def flaky_embed(texts):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("embedding backend down")
        return [[0.0] for _ in texts]

    checkpoint = _run_ingestion_with_two_batches(tmp_path, monkeypatch, flaky_embed)

    assert calls["n"] >= 2
    assert checkpoint["processed_files"] == []


def test_run_ingestion_all_batches_embedded_marks_filing_processed(tmp_path, monkeypatch):
    checkpoint = _run_ingestion_with_two_batches(tmp_path, monkeypatch, lambda texts: [[0.0] for _ in texts])

    assert checkpoint["processed_files"] == ["0000000000-24-000001"]
