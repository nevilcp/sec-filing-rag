"""
rag/edgar_client.py

EDGAR HTTP client for filing discovery and document acquisition.

Responsibilities:
  - Query EDGAR submissions API for recent filings
  - Download primary filing documents
  - Enforce global rate limiting for SEC domain compliance

Dependencies:
  - httpx
  - tenacity
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from src.core import config
from src.core.corpus import _get_user_agent
from src.core.fileio import atomic_write_text
from src.core.logging import get_logger

logger = get_logger(__name__)

# HTTP statuses that signal transient SEC throttling.
_RETRYABLE_STATUS = (429, 503)

_LAST_REQUEST_TIME: float = 0.0
_THROTTLE_LOCK = threading.Lock()


def _throttle() -> None:
    """Enforces EDGAR_RATE_LIMIT_REQUESTS_PER_SECOND using simple sleep"""
    global _LAST_REQUEST_TIME
    with _THROTTLE_LOCK:
        now = time.time()
        elapsed = now - _LAST_REQUEST_TIME
        delay = (1.0 / config.EDGAR_RATE_LIMIT_REQUESTS_PER_SECOND) - elapsed
        if delay > 0:
            time.sleep(delay)
        _LAST_REQUEST_TIME = time.time()


def _is_retryable_http_error(exc: BaseException) -> bool:
    """Return True for HTTP 429 and 503 errors from the SEC EDGAR API.

    Both codes indicate transient server-side throttling that should
    be retried with exponential backoff rather than propagated immediately.

    Args:
        exc: The exception raised by the HTTP call.

    Returns:
        True if the error warrants a retry.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRYABLE_STATUS
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
def _edgar_get(url: str) -> httpx.Response:
    """Throttle, then GET a URL from EDGAR with the required headers.

    Decorated with tenacity retry logic for 429/503 responses. Throttling
    is applied before each HTTP call to respect the SEC's request-rate
    policy rather than relying solely on reactive backoff.

    Args:
        url: The fully-qualified EDGAR URL to fetch.

    Returns:
        The successful HTTP response.

    Raises:
        httpx.HTTPStatusError: On non-retryable HTTP errors.
    """
    _throttle()
    headers = {"User-Agent": _get_user_agent(), "Accept-Encoding": "gzip, deflate"}
    logger.debug("Fetching EDGAR URL %s", url)
    with httpx.Client(timeout=60.0) as client:
        response = client.get(url, headers=headers)
        response.raise_for_status()
        return response


def _edgar_get_json(url: str) -> dict:
    """Fetch a JSON document from an EDGAR endpoint and parse it.

    Args:
        url: The fully-qualified EDGAR API URL to fetch.

    Returns:
        The parsed JSON response as a Python dict.
    """
    return _edgar_get(url).json()


def _edgar_get_text(url: str) -> str:
    """Fetch a raw text document from an EDGAR endpoint.

    Args:
        url: The fully-qualified EDGAR document URL to fetch.

    Returns:
        The raw response body as a string.
    """
    return _edgar_get(url).text


def get_recent_filings(cik: str, form_type: str = "10-K", max_results: int = 1) -> list[dict]:
    """Queries the submissions endpoint for recent filings matching the target fiscal years.

    Filters results by ``config.EDGAR_FISCAL_YEARS``, comparing against the
    ``period_of_report`` field (fiscal year-end date, e.g. '2024-12-31') returned
    by the EDGAR submissions API.

    Args:
        cik: The company's Central Index Key.
        form_type: Target filing form (e.g., '10-K').
        max_results: Maximum number of recent filings to return.

    Returns:
        A list of filing metadata dictionaries whose fiscal year matches
        any of the years in ``config.EDGAR_FISCAL_YEARS``
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
    for i in range(len(accession_numbers)):
        if forms[i] != form_type:
            continue

        if form_type not in config.EDGAR_TARGET_FORM_TYPES:
            continue

        period = report_dates[i] if i < len(report_dates) else ""

        if not any(period.startswith(target) for target in config.EDGAR_FISCAL_YEARS):
            continue

        results.append(
            {
                "accession_number": accession_numbers[i],
                "filing_date": filing_dates[i],
                "period_of_report": period,
                "form_type": forms[i],
                "primary_document": primary_documents[i],
            }
        )
        if len(results) >= max_results:
            break

    return results


def download_primary_document(cik: str, accession_number: str, filename: str) -> str:
    """Download the raw HTML content of a filing's primary document.

    Args:
        cik: The company's zero-padded 10-digit CIK.
        accession_number: The filing's unique accession number.
        filename: The primary document filename from the filing index.

    Returns:
        The raw HTML content of the document as a string.
    """
    accession_clean = accession_number.replace("-", "")
    cik_clean = str(int(cik))

    url = f"https://www.sec.gov/Archives/edgar/data/{cik_clean}/{accession_clean}/{filename}"
    return _edgar_get_text(url)


def stage_filing(cik: str, ticker: str, accession_number: str, filename: str, staging_dir: Path) -> Path:
    """Downloads the primary document and writes it to the staging directory.

    Idempotent operation: skips download if the target file already exists.

    Args:
        cik: The company's CIK.
        ticker: The company ticker symbol (used for folder organization).
        accession_number: The filing's unique identifier.
        filename: The primary document's filename from the filing index.
        staging_dir: Root directory for saving downloaded HTML files.

    Returns:
        The absolute path to the staged HTML file
    """
    ticker_dir = staging_dir / ticker.upper()
    ticker_dir.mkdir(parents=True, exist_ok=True)

    file_path = ticker_dir / f"{accession_number}.html"

    if file_path.exists():
        logger.info("Filing already staged at %s, skipping download.", file_path)
        return file_path

    logger.info("Downloading primary document for %s (%s)", ticker, accession_number)
    content = download_primary_document(cik, accession_number, filename)

    atomic_write_text(file_path, content)

    logger.info("Staged filing to %s", file_path)
    return file_path
