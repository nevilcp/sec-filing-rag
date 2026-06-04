"""
tests/unit/test_section_extractor.py

Unit tests for the multi-tier EDGAR HTML section extractor.

Responsibilities:
  - Validate anchor-based, heading-based, and fallback extraction logic
  - Ensure structural noise (tables, XBRL tags, short lines) is purged
  - Test the full parsing pipeline against varied document structures
"""

from bs4 import BeautifulSoup

from src.rag.section_extractor import (
    _clean_extracted_text,
    _extract_tier_1,
    _extract_tier_2,
    _extract_tier_3,
    extract_sections,
)


def test_clean_extracted_text():
    raw_html = """
    <div>
        <p>Short line</p>
        <p>This is a valid sentence that should absolutely be kept in the extraction because it is long enough.</p>
        <p>12,345.67</p>
        <p>-------</p>
        <p>Another valid sentence that provides meaningful context for the financial analysis.</p>
    </div>
    """
    cleaned = _clean_extracted_text(raw_html)
    assert "Short line" not in cleaned
    assert "12,345.67" not in cleaned
    assert "-------" not in cleaned
    assert (
        "This is a valid sentence that should absolutely be kept in the extraction because it is long enough."
        in cleaned
    )
    assert "Another valid sentence that provides meaningful context for the financial analysis." in cleaned
    assert cleaned.count("\n") == 1


def test_extract_tier_1_anchor():
    html = """
    <html>
        <body>
            <a id="item1"></a>
            <p>This is the content for item 1. It is very long and has lots of details about the business.</p>
            <a id="item1a"></a>
            <p>This is the content for item 1A.</p>
        </body>
    </html>
    """
    soup = BeautifulSoup(html, "lxml")
    res = _extract_tier_1(soup, "Item 1")
    assert "This is the content for item 1" in res
    assert "This is the content for item 1A" not in res


def test_extract_tier_2_heading():
    html = """
    <html>
        <body>
            <h2>Item 1. Business</h2>
            <p>This is the business section content. We are a great company.</p>
            <h2>Item 1A. Risk Factors</h2>
            <p>These are the risks. Beware!</p>
        </body>
    </html>
    """
    soup = BeautifulSoup(html, "lxml")
    res = _extract_tier_2(soup, "Item 1")
    assert "This is the business section content" in res
    assert "These are the risks" not in res


def test_extract_tier_3_regex():
    text = """
    Table of Contents
    Item 1. Business .... page 4
    Item 1A. Risk Factors .... page 10

    Item 1. Business
    This is the actual business section. It contains a lot of text that describes what we do and how we make money.
    It is much longer than the table of contents entry.

    Item 1A. Risk Factors
    We might lose money
    """
    res = _extract_tier_3(text, "Item 1")
    assert res is not None
    assert "This is the actual business section" in res
    assert "Table of Contents" not in res
    assert "Item 1A" not in res


def test_extract_sections_integration():
    html = """
    <html>
        <body>
            <ix:nonnumeric name="us-gaap:BusinessDescription">
                Hidden XBRL stuff
            </ix:nonnumeric>
            <h2>Item 1. Business</h2>
            <p>This is a valid sentence that should absolutely be kept in the extraction because it is long enough.</p>
            <p>Another valid sentence that provides meaningful context for the financial analysis.</p>
            <p>And here is a third valid sentence to ensure the length is well over 100 characters so it is not skipped.</p>
            <h2>Item 1A. Risk Factors</h2>
        </body>
    </html>
    """
    sections = extract_sections(html)
    assert "business_overview" in sections
    assert "Hidden XBRL stuff" not in sections["business_overview"]
    assert "This is a valid sentence" in sections["business_overview"]


def test_extract_sections_fallback():
    html = """
    <html>
        <body>
            <p>No standard headings are present.</p>
            <p>This is a valid sentence that should absolutely be kept in the extraction because it is long enough.</p>
            <p>Another valid sentence that provides meaningful context for the financial analysis.</p>
            <p>And here is a third valid sentence to ensure the length is well over 100 characters so it is not skipped.</p>
        </body>
    </html>
    """
    sections = extract_sections(html)
    assert "full_text" in sections
    assert "This is a valid sentence" in sections["full_text"]
