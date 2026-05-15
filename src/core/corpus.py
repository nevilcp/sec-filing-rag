"""Company identity layer for the EDGAR ingestion pipeline."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import httpx

from src.core import config
from src.core.constants import COMPANY_MANIFEST, KNOWN_TICKERS
from src.core.logging import get_logger

logger = get_logger(__name__)

SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
CACHE_MAX_AGE_SECONDS = 7 * 24 * 60 * 60  # 7 days

_IN_MEMORY_CIK_MAP: dict[str, dict] | None = None


def _get_user_agent() -> str:
    email = os.getenv("EDGAR_CONTACT_EMAIL", "contact@example.com")
    return f"ComplianceRAG {email}"


def get_cik_map() -> dict[str, dict]:
    """
    Downloads or loads the EDGAR bulk company-tickers JSON.
    Caches the result to disk, and uses the cache if it's <= 7 days old.
    Returns the raw JSON dict.
    """
    global _IN_MEMORY_CIK_MAP
    if _IN_MEMORY_CIK_MAP is not None:
        return _IN_MEMORY_CIK_MAP

    cache_path = Path(config.CIK_CACHE_PATH)

    # Check if cache is valid (exists and <= 7 days old)
    if cache_path.exists():
        mtime = cache_path.stat().st_mtime
        if (time.time() - mtime) <= CACHE_MAX_AGE_SECONDS:
            logger.info("Loading CIK map from cache: %s", cache_path)
            with open(cache_path, "r", encoding="utf-8") as f:
                _IN_MEMORY_CIK_MAP = json.load(f)
                return _IN_MEMORY_CIK_MAP

    logger.info("Downloading CIK map from %s", SEC_COMPANY_TICKERS_URL)
    headers = {"User-Agent": _get_user_agent()}

    with httpx.Client(timeout=30.0) as client:
        response = client.get(SEC_COMPANY_TICKERS_URL, headers=headers)
        response.raise_for_status()
        data = response.json()

    # Ensure parent directory exists before writing cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    logger.info("Cached CIK map to %s", cache_path)
    _IN_MEMORY_CIK_MAP = data
    return data


def resolve_cik(ticker: str) -> str:
    """
    Looks up a ticker string in the CIK map.
    Returns the zero-padded 10-digit CIK string.
    Raises ValueError if the ticker is not found.
    """
    ticker_upper = ticker.upper()
    cik_map = get_cik_map()

    for entry in cik_map.values():
        if entry.get("ticker", "").upper() == ticker_upper:
            cik = str(entry.get("cik_str"))
            return cik.zfill(10)

    raise ValueError(f"Ticker '{ticker}' not found in EDGAR company-tickers map.")


def get_corpus_companies() -> list[dict]:
    """
    Returns COMPANY_MANIFEST enriched with a resolved 'cik' field.
    Raises ValueError if any ticker cannot be resolved.
    """
    enriched_manifest = []

    # Create a quick lookup dict for efficiency
    cik_map = get_cik_map()
    ticker_to_cik = {
        entry.get("ticker", "").upper(): str(entry.get("cik_str")).zfill(10)
        for entry in cik_map.values()
    }

    for company in COMPANY_MANIFEST:
        ticker = company["ticker"].upper()
        if ticker not in ticker_to_cik:
            raise ValueError(f"Ticker '{ticker}' from COMPANY_MANIFEST not found in EDGAR company-tickers map.")

        enriched_company = company.copy()
        enriched_company["cik"] = ticker_to_cik[ticker]
        enriched_manifest.append(enriched_company)

    return enriched_manifest


def is_known_ticker(ticker: str) -> bool:
    """
    Returns True if the ticker exists in KNOWN_TICKERS (case-insensitive).
    """
    ticker_upper = ticker.upper()
    return any(t.upper() == ticker_upper for t in KNOWN_TICKERS)
