"""Unit tests for the Gradio citation panel."""

from unittest import mock

from app.gradio_app import build_citations_html
from src.core import config


def test_build_citations_html_quote_in_accession_stays_inside_href():
    html_out = build_citations_html([{"accession_number": '0000320193-24-000123" onmouseover="x', "ticker": "AAPL"}])

    assert 'onmouseover="x' not in html_out
    assert "&quot;" in html_out


def test_citation_uses_issuer_cik_instead_of_filing_agent():
    html_out = build_citations_html(
        [{"accession_number": "0000950170-25-100235", "ticker": "MSFT", "cik": "0000789019"}]
    )

    assert "/edgar/data/789019/000095017025100235/0000950170-25-100235-index.htm" in html_out


@mock.patch("app.gradio_app.requests.post")
def test_query_timeout_allows_provider_pacing(mock_post):
    from app.gradio_app import submit_query

    mock_post.return_value.status_code = 500
    submit_query("Supply chain risks?", [], "session", "AAPL")
    assert mock_post.call_args.kwargs["timeout"] == config.QUERY_TIMEOUT_SECONDS
