"""Company identity layer for the EDGAR ingestion pipeline."""

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
    email = os.getenv("EDGAR_CONTACT_EMAIL")
    if not email:
        raise ValueError(
            "CRITICAL: 'EDGAR_CONTACT_EMAIL' environment variable is missing. "
            "The SEC requires a real email address in the User-Agent header to avoid IP blocks."
        )
    return f"ComplianceRAG {email}"


def get_cik_map() -> dict[str, dict]:
    """Load the SEC ticker map from memory, a seven-day disk cache, or the SEC endpoint."""
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
    """Resolve a ticker to its zero-padded 10-digit CIK, or raise if it is unknown."""
    ticker_upper = ticker.upper()
    ticker_to_cik = {c["ticker"].upper(): c["cik"] for c in get_corpus_companies()}
    if ticker_upper not in ticker_to_cik:
        raise ValueError(f"Ticker '{ticker}' not found in EDGAR company-tickers map.")
    return ticker_to_cik[ticker_upper]


def get_corpus_companies() -> list[dict]:
    """Resolve the configured corpus tickers to SEC company identities."""
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
