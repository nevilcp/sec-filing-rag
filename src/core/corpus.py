"""Company identity layer for the EDGAR ingestion pipeline.

Responsibilities:
  - Fetch and cache the SEC EDGAR company-tickers map.
  - Resolve ticker symbols to zero-padded 10-digit CIKs.
  - Enrich KNOWN_TICKERS with CIKs and company names for ingestion.
  - Build the SEC-required User-Agent header value.

Dependencies:
  - EDGAR_CONTACT_EMAIL must be set before any SEC request.
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

# Seven days, in seconds.
CACHE_MAX_AGE_SECONDS = 7 * 24 * 60 * 60

_IN_MEMORY_CIK_MAP: dict[str, dict] | None = None


def _get_user_agent() -> str:
    """Build the SEC-required User-Agent header value.

    The SEC requires a real email address in the User-Agent to identify
    the caller and avoid IP-based throttling or blocks.

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
    """Return the EDGAR bulk company-tickers map.

    Reads a disk cache when the file is younger than 7 days, to avoid
    rate limiting by the SEC endpoint. Otherwise downloads a fresh copy
    and rewrites the cache. The result is also kept in memory.

    Returns:
        A dict mapping a string index to company details, for example
        {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "..."}}.

    Raises:
        ValueError: If EDGAR_CONTACT_EMAIL is not set, or the cache file
            holds invalid JSON.
        httpx.HTTPStatusError: If the SEC endpoint returns an error
            status.
        httpx.RequestError: If the request fails at the network level.
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
    """Resolve a ticker symbol to its zero-padded 10-digit CIK.

    Args:
        ticker: The stock ticker symbol, case-insensitive.

    Returns:
        The 10-digit CIK string used for EDGAR lookups.

    Raises:
        ValueError: If the ticker is not in the corpus, or if the
            company map cannot be built (see get_corpus_companies).
    """
    ticker_upper = ticker.upper()
    ticker_to_cik = {c["ticker"].upper(): c["cik"] for c in get_corpus_companies()}
    if ticker_upper not in ticker_to_cik:
        raise ValueError(f"Ticker '{ticker}' not found in EDGAR company-tickers map.")
    return ticker_to_cik[ticker_upper]


def get_corpus_companies() -> list[dict]:
    """Build the company manifest from KNOWN_TICKERS using EDGAR data.

    Matches each ticker against the EDGAR company-tickers map to find
    the official CIK and company name.

    Returns:
        One dict per ticker, with 'ticker', 'company_name' and a
        zero-padded 'cik' string.

    Raises:
        ValueError: If a ticker cannot be found in the EDGAR map.
        httpx.HTTPError: If the company map must be downloaded and the
            request fails (see get_cik_map).
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
