"""EDGAR HTTP client for filing discovery and document acquisition."""

from __future__ import annotations

import os
import time
from pathlib import Path

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from src.core import config
from src.core.logging import get_logger

logger = get_logger(__name__)

_LAST_REQUEST_TIME: float = 0.0


def _throttle() -> None:
    """Enforces EDGAR_RATE_LIMIT_REQUESTS_PER_SECOND using simple sleep."""
    global _LAST_REQUEST_TIME
    now = time.time()
    elapsed = now - _LAST_REQUEST_TIME
    delay = (1.0 / config.EDGAR_RATE_LIMIT_REQUESTS_PER_SECOND) - elapsed
    if delay > 0:
        time.sleep(delay)
    _LAST_REQUEST_TIME = time.time()


def _get_user_agent() -> str:
    email = os.getenv("EDGAR_CONTACT_EMAIL", "contact@example.com")
    return f"ComplianceRAG {email}"


def _is_retryable_http_error(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 503)
    return False


@retry(
    reraise=True,
    retry=retry_if_exception(_is_retryable_http_error),
    wait=wait_exponential(
        multiplier=config.RETRY_MULTIPLIER_SECONDS,
        min=config.RETRY_MIN_WAIT_SECONDS,
        max=config.RETRY_MAX_WAIT_SECONDS,
    ),
    stop=stop_after_attempt(config.RETRY_MAX_ATTEMPTS),
)
def _edgar_get_json(url: str) -> dict:
    _throttle()
    headers = {"User-Agent": _get_user_agent(), "Accept-Encoding": "gzip, deflate"}
    logger.debug("Fetching EDGAR JSON from %s", url)
    with httpx.Client(timeout=60.0) as client:
        response = client.get(url, headers=headers)
        response.raise_for_status()
        return response.json()


@retry(
    reraise=True,
    retry=retry_if_exception(_is_retryable_http_error),
    wait=wait_exponential(
        multiplier=config.RETRY_MULTIPLIER_SECONDS,
        min=config.RETRY_MIN_WAIT_SECONDS,
        max=config.RETRY_MAX_WAIT_SECONDS,
    ),
    stop=stop_after_attempt(config.RETRY_MAX_ATTEMPTS),
)
def _edgar_get_text(url: str) -> str:
    _throttle()
    headers = {"User-Agent": _get_user_agent(), "Accept-Encoding": "gzip, deflate"}
    logger.debug("Fetching EDGAR text from %s", url)
    with httpx.Client(timeout=60.0) as client:
        response = client.get(url, headers=headers)
        response.raise_for_status()
        return response.text


def get_recent_filings(cik: str, form_type: str = "10-K", max_results: int = 1) -> list[dict]:
    """
    Queries the submissions endpoint for recent filings.
    Returns up to max_results entries filtered by form_type.
    """
    url = f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json"
    data = _edgar_get_json(url)

    filings = data.get("filings", {}).get("recent", {})
    if not filings:
        return []

    accession_numbers = filings.get("accessionNumber", [])
    filing_dates = filings.get("filingDate", [])
    report_dates = filings.get("reportDate", [])
    forms = filings.get("form", [])
    primary_documents = filings.get("primaryDocument", [])

    results = []
    # Iterate through parallel arrays (newest first)
    for i in range(len(accession_numbers)):
        if forms[i] == form_type:
            results.append(
                {
                    "accession_number": accession_numbers[i],
                    "filing_date": filing_dates[i],
                    "period_of_report": report_dates[i] if i < len(report_dates) else "",
                    "form_type": forms[i],
                    "primary_document": primary_documents[i],
                }
            )
            if len(results) >= max_results:
                break

    return results


def get_filing_index(cik: str, accession_number: str) -> list[dict]:
    """
    Fetches the JSON index for a specific filing.
    Returns a list of dicts describing each file in the index.
    """
    accession_clean = accession_number.replace("-", "")
    cik_clean = str(int(cik))  # SEC Archives URLs typically strip leading zeros from CIK

    base_url = f"https://www.sec.gov/Archives/edgar/data/{cik_clean}/{accession_clean}"
    url = f"{base_url}/{accession_clean}-index.json"

    data = _edgar_get_json(url)
    items = data.get("directory", {}).get("item", [])

    results = []
    for item in items:
        file_url = f"{base_url}/{item.get('name')}"
        results.append(
            {
                "name": item.get("name"),
                "type": item.get("type"),
                "description": item.get("description"),
                "url": file_url,
            }
        )
    return results


def download_primary_document(cik: str, accession_number: str, filename: str) -> str:
    """
    Downloads and returns the raw text content of a primary document.
    """
    accession_clean = accession_number.replace("-", "")
    cik_clean = str(int(cik))

    url = f"https://www.sec.gov/Archives/edgar/data/{cik_clean}/{accession_clean}/{filename}"
    return _edgar_get_text(url)


def stage_filing(
    cik: str, ticker: str, accession_number: str, filename: str, staging_dir: Path
) -> Path:
    """
    Downloads the primary document and writes it to the staging directory.
    Returns the Path to the staged file. Idempotent if the file exists.
    """
    ticker_dir = staging_dir / ticker.upper()
    ticker_dir.mkdir(parents=True, exist_ok=True)

    file_path = ticker_dir / f"{accession_number}.html"

    if file_path.exists():
        logger.info("Filing already staged at %s, skipping download.", file_path)
        return file_path

    logger.info("Downloading primary document for %s (%s)", ticker, accession_number)
    content = download_primary_document(cik, accession_number, filename)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info("Staged filing to %s", file_path)
    return file_path
