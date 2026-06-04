"""
core/corpus.py

Company identity layer for the EDGAR ingestion pipeline.

Responsibilities:
  - Fetch and cache the SEC EDGAR company-tickers map
  - Resolve ticker symbols to their zero-padded 10-digit CIKs
  - Enrich the target KNOWN_TICKERS with CIKs and company names for downstream ingestion

Dependencies:
  - httpx
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import httpx

from src.core import config
from src.core.constants import KNOWN_TICKERS
from src.core.logging import get_logger

logger = get_logger(__name__)

SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

CACHE_MAX_AGE_SECONDS = 7 * 24 * 60 * 60

_IN_MEMORY_CIK_MAP: dict[str, dict] | None = None


def _get_user_agent() -> str:
    """Build the SEC-required User-Agent header value.

    Returns:
        A User-Agent string in the format 'ComplianceRAG <email>'.

    Raises:
        ValueError: If EDGAR_CONTACT_EMAIL is not set.
    """
    email = os.getenv("EDGAR_CONTACT_EMAIL")
    if not email:
        raise ValueError(
            "CRITICAL: 'EDGAR_CONTACT_EMAIL' environment variable is missing. "
            "The SEC requires a real email address in the User-Agent header to avoid IP blocks."
        )
    return f"ComplianceRAG {email}"


def get_cik_map() -> dict[str, dict]:
    """Retrieves the EDGAR bulk company-tickers JSON.

    Uses a disk-backed cache to avoid rate-limiting by the SEC endpoint.
    Loads from cache if the file is younger than 7 days, otherwise issues
    a fresh HTTP request.

    Returns:
        A dictionary mapping a string index to company details
        (e.g., {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}).

    Raises:
        httpx.HTTPStatusError: If the SEC endpoint returns a non-200 status
    """
    global _IN_MEMORY_CIK_MAP
    if _IN_MEMORY_CIK_MAP is not None:
        return _IN_MEMORY_CIK_MAP

    cache_path = Path(config.CIK_CACHE_PATH)
    if cache_path.exists():
        mtime = cache_path.stat().st_mtime
        if (time.time() - mtime) <= CACHE_MAX_AGE_SECONDS:
            logger.info("Loading CIK map from cache: %s", cache_path)
            with cache_path.open("r", encoding="utf-8") as f:
                _IN_MEMORY_CIK_MAP = json.load(f)
                return _IN_MEMORY_CIK_MAP

    logger.info("Downloading CIK map from %s", SEC_COMPANY_TICKERS_URL)
    headers = {"User-Agent": _get_user_agent()}

    with httpx.Client(timeout=30.0) as client:
        response = client.get(SEC_COMPANY_TICKERS_URL, headers=headers)
        response.raise_for_status()
        data = response.json()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(data, f)

    logger.info("Cached CIK map to %s", cache_path)
    _IN_MEMORY_CIK_MAP = data
    return data


def resolve_cik(ticker: str) -> str:
    """Resolves a ticker symbol to its zero-padded 10-digit CIK string.

    Args:
        ticker: The stock ticker symbol (case-insensitive).

    Returns:
        The zero-padded 10-digit CIK string required for EDGAR archival lookups.

    Raises:
        ValueError: If the ticker is not found in the enriched corpus manifest
    """
    ticker_upper = ticker.upper()
    ticker_to_cik = {c["ticker"].upper(): c["cik"] for c in get_corpus_companies()}
    if ticker_upper not in ticker_to_cik:
        raise ValueError(f"Ticker '{ticker}' not found in EDGAR company-tickers map.")
    return ticker_to_cik[ticker_upper]


def get_corpus_companies() -> list[dict]:
    """Dynamically builds the company manifest from KNOWN_TICKERS using EDGAR data.

    Matches each ticker against the EDGAR company-tickers map to retrieve
    the official CIK, returning a complete manifest for the ingestion pipeline.

    Returns:
        A list of dictionaries, where each dict contains the original
        company metadata plus the resolved 'cik' string.

    Raises:
        ValueError: If any ticker in the manifest cannot be resolved via EDGAR
    """
    enriched_manifest = []

    cik_map = get_cik_map()
    ticker_to_entry = {entry.get("ticker", "").upper(): entry for entry in cik_map.values()}

    for ticker in KNOWN_TICKERS:
        ticker = ticker.upper()
        if ticker not in ticker_to_entry:
            raise ValueError(f"Ticker '{ticker}' from KNOWN_TICKERS not found in EDGAR company-tickers map.")

        entry = ticker_to_entry[ticker]
        enriched_manifest.append(
            {
                "ticker": ticker,
                "company_name": entry.get("title"),
                "cik": str(entry.get("cik_str")).zfill(10),
            }
        )

    return enriched_manifest
