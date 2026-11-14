"""Unit tests for the EDGAR HTTP client helpers."""

from __future__ import annotations

from unittest import mock

import pytest

from src.rag.edgar_client import _get_user_agent


def test_get_user_agent_raises_without_email(monkeypatch):
    monkeypatch.delenv("EDGAR_CONTACT_EMAIL", raising=False)
    with pytest.raises(ValueError, match="EDGAR_CONTACT_EMAIL"):
        _get_user_agent()


def test_get_user_agent_returns_formatted_string(monkeypatch):
    monkeypatch.setenv("EDGAR_CONTACT_EMAIL", "test@example.com")
    agent = _get_user_agent()
    assert "test@example.com" in agent
    assert "ComplianceRAG" in agent


def _make_filings_payload(periods: list[str], form_type: str = "10-K") -> dict:
    n = len(periods)
    return {
        "filings": {
            "recent": {
                "accessionNumber": [f"0001234567-24-{i:06d}" for i in range(n)],
                "filingDate": ["2024-01-01"] * n,
                "reportDate": periods,
                "form": [form_type] * n,
                "primaryDocument": [f"filing{i}.htm" for i in range(n)],
            }
        }
    }


@mock.patch("src.rag.edgar_client._edgar_get_json")
def test_get_recent_filings_filters_by_fiscal_year(mock_get_json):
    mock_get_json.return_value = _make_filings_payload(["2023-12-31", "2024-12-31", "2022-12-31"])

    with mock.patch("src.core.config.EDGAR_FISCAL_YEARS", ["2024"]):
        from src.rag import edgar_client

        results = edgar_client.get_recent_filings("0000320193")

    assert len(results) == 1
    assert results[0]["period_of_report"] == "2024-12-31"


@mock.patch("src.rag.edgar_client._edgar_get_json")
def test_get_recent_filings_multi_year(mock_get_json):
    mock_get_json.return_value = _make_filings_payload(["2023-12-31", "2024-12-31", "2021-12-31"])

    with mock.patch("src.core.config.EDGAR_FISCAL_YEARS", ["2023", "2024"]):
        from src.rag import edgar_client

        results = edgar_client.get_recent_filings("0000320193", max_results=5)

    assert len(results) == 2
    periods = {r["period_of_report"] for r in results}
    assert periods == {"2023-12-31", "2024-12-31"}


@mock.patch("src.rag.edgar_client._edgar_get_json")
def test_get_recent_filings_wrong_form_type_excluded(mock_get_json):
    mock_get_json.return_value = _make_filings_payload(["2024-12-31"], form_type="10-Q")

    with mock.patch("src.core.config.EDGAR_FISCAL_YEARS", ["2024"]):
        from src.rag import edgar_client

        results = edgar_client.get_recent_filings("0000320193", form_type="10-K")

    assert results == []


@mock.patch("src.rag.edgar_client._edgar_get_json")
def test_get_recent_filings_empty_response(mock_get_json):
    mock_get_json.return_value = {}
    from src.rag import edgar_client

    results = edgar_client.get_recent_filings("0000320193")
    assert results == []
