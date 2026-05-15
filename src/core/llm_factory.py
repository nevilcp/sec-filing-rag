"""Centralized LLM factory functions with cached instances and retry policy."""

from __future__ import annotations

import os
from typing import Final

from langchain_google_genai import ChatGoogleGenerativeAI
from openinference.instrumentation.langchain import LangChainInstrumentor
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from src.core import config

if os.getenv("PHOENIX_ENABLED", "").lower() == "true":
    try:
        from phoenix.otel import register
        register(endpoint="http://localhost:6006/v1/traces")
        LangChainInstrumentor().instrument()
    except Exception as e:
        pass

_PRIMARY_LLM: ChatGoogleGenerativeAI | None = None
_PREMIUM_LLM: ChatGoogleGenerativeAI | None = None
_CLASSIFICATION_LLM: ChatGoogleGenerativeAI | None = None

_RATE_LIMIT_ERROR_MARKERS: Final[tuple[str, ...]] = (
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
            "Define GEMINI_API_KEY in your environment before creating LLM clients."
        )
    return api_key


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
def _build_llm(api_key: str, model_name: str, temperature: float) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=temperature,
    )


def get_primary_llm() -> ChatGoogleGenerativeAI:
    global _PRIMARY_LLM
    if _PRIMARY_LLM is None:
        _PRIMARY_LLM = _build_llm(
            _require_gemini_api_key(),
            config.GENERATION_PRIMARY_MODEL,
            config.ROUTING_LLM_TEMPERATURE,
        )
    return _PRIMARY_LLM


def get_premium_llm() -> ChatGoogleGenerativeAI:
    global _PREMIUM_LLM
    if _PREMIUM_LLM is None:
        _PREMIUM_LLM = _build_llm(
            _require_gemini_api_key(),
            config.GENERATION_PREMIUM_MODEL,
            config.GENERATION_LLM_TEMPERATURE,
        )
    return _PREMIUM_LLM


def get_classification_llm() -> ChatGoogleGenerativeAI:
    global _CLASSIFICATION_LLM
    if _CLASSIFICATION_LLM is None:
        _CLASSIFICATION_LLM = _build_llm(
            _require_gemini_api_key(),
            config.CLASSIFICATION_MODEL,
            0.0,
        )
    return _CLASSIFICATION_LLM


def get_router_llm() -> ChatGoogleGenerativeAI:
    """Alias for get_primary_llm() for backward compatibility."""
    return get_primary_llm()


def get_generation_llm() -> ChatGoogleGenerativeAI:
    """Alias for get_premium_llm() for backward compatibility."""
    return get_premium_llm()
