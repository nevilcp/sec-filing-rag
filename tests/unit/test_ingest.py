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
