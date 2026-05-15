"""Centralized embedding client with rate-limited retry wrappers.

Uses the ``google-genai`` SDK directly (instead of the LangChain wrapper)
to correctly batch multiple texts into a single ``embed_content`` call that
returns one embedding per input text.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

from google import genai
from google.genai.types import Content, Part
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from src.core import config
from src.core.logging import get_logger
from src.core.rate_limiter import estimate_tokens, get_embedding_rate_limiter

logger = get_logger(__name__)

_CLIENT: genai.Client | None = None

_RATE_LIMIT_ERROR_MARKERS: tuple[str, ...] = (
    "rate limit",
    "too many requests",
    "resource exhausted",
    "resource_exhausted",
    "quota",
    "429",
)


def _is_rate_limit_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in _RATE_LIMIT_ERROR_MARKERS)


def _require_gemini_api_key() -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is required but not set. "
            "Define GEMINI_API_KEY in your environment before creating embedding clients."
        )
    return api_key


def _get_client() -> genai.Client:
    """Return cached genai Client."""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = genai.Client(api_key=_require_gemini_api_key())
    return _CLIENT


@retry(
    reraise=True,
    retry=retry_if_exception(_is_rate_limit_error),
    wait=wait_exponential(
        multiplier=config.RETRY_MULTIPLIER_SECONDS,
        min=config.RETRY_MIN_WAIT_SECONDS,
        max=config.RETRY_MAX_WAIT_SECONDS,
    ),
    stop=stop_after_attempt(config.RETRY_MAX_ATTEMPTS),
)
def embed_query(text: str) -> list[float]:
    """Embed a single query string with rate limiting and retry."""
    limiter = get_embedding_rate_limiter()
    tokens = estimate_tokens([text])
    limiter.acquire(tokens)

    client = _get_client()
    response = client.models.embed_content(
        model=config.EMBEDDING_MODEL_NAME,
        contents=Content(parts=[Part(text=text)]),
    )
    return list(response.embeddings[0].values)


@retry(
    reraise=True,
    retry=retry_if_exception(_is_rate_limit_error),
    wait=wait_exponential(
        multiplier=config.RETRY_MULTIPLIER_SECONDS,
        min=config.RETRY_MIN_WAIT_SECONDS,
        max=config.RETRY_MAX_WAIT_SECONDS,
    ),
    stop=stop_after_attempt(config.RETRY_MAX_ATTEMPTS),
)
def embed_documents(texts: Sequence[str]) -> list[list[float]]:
    """Embed multiple documents in a single API call with rate limiting and retry.

    Each text is wrapped in its own ``Content`` object so the API returns
    one embedding vector per input text.
    """
    limiter = get_embedding_rate_limiter()
    tokens = estimate_tokens(texts)
    limiter.acquire(tokens)

    client = _get_client()
    contents = [Content(parts=[Part(text=t)]) for t in texts]
    response = client.models.embed_content(
        model=config.EMBEDDING_MODEL_NAME,
        contents=contents,
    )
    return [list(emb.values) for emb in response.embeddings]
