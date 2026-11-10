"""
Unit tests for the Gradio citation panel.

Responsibilities:
  - Verify values from the API are escaped before they reach HTML.
"""

from app.gradio_app import build_citations_html


def test_build_citations_html_quote_in_accession_stays_inside_href():
    html_out = build_citations_html([{"accession_number": '0000320193-24-000123" onmouseover="x', "ticker": "AAPL"}])

    assert 'onmouseover="x' not in html_out
    assert "&quot;" in html_out
