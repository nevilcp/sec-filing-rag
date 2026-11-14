"""Centralized LLM factories with cached instances and a retry policy."""

from __future__ import annotations

import os
from typing import Any, Final

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from tenacity import (
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from src.core import config
from src.core.lazy import locked_singleton
from src.core.rate_limiter import (
    _SlidingWindowRateLimiter,
    estimate_llm_tokens,
)

_RATE_LIMIT_ERROR_MARKERS: Final[tuple[str, ...]] = (
    "rate limit",
    "too many requests",
    "resource exhausted",
    "resource_exhausted",
    "quota",
    "429",
)


def _is_rate_limit_error(exc: BaseException) -> bool:
    """Recognize transient rate-limit errors while excluding daily quota exhaustion."""
    message = str(exc).lower()

    if any(marker in message for marker in ("tokens per day", "tpd", "requests per day", "rpd")):
        return False

    return any(marker in message for marker in _RATE_LIMIT_ERROR_MARKERS)


def _require_groq_api_key() -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is required but not set. Define GROQ_API_KEY in your environment before creating LLM clients."
        )
    return api_key


def _require_gemini_api_key() -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is required but not set. "
            "Define GEMINI_API_KEY in your environment "
            "before creating Gemini clients."
        )
    return api_key


def _build_groq_llm(
    api_key: str,
    model_name: str,
    temperature: float,
    *,
    reasoning_effort: str | None = None,
    reasoning_format: str | None = None,
) -> ChatGroq:
    kwargs: dict = {
        "model": model_name,
        "api_key": api_key,
        "temperature": temperature,
        "max_retries": config.LLM_SDK_MAX_RETRIES,
    }
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort
    if reasoning_format is not None:
        kwargs["reasoning_format"] = reasoning_format
    return ChatGroq(**kwargs)


def _build_gemini_llm(
    api_key: str,
    model_name: str,
    temperature: float,
) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=temperature,
        max_retries=config.LLM_SDK_MAX_RETRIES,
    )


def _extract_content_str(content: object) -> str:
    """Normalize plain text and LangChain content blocks to a single string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text" and "text" in block:
                    parts.append(block["text"])
                elif "text" in block:
                    parts.append(str(block["text"]))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content)


def _do_invoke_with_retry(llm: BaseChatModel, prompt: str) -> Any:
    """Retry transient per-minute throttling with backoff and propagate other errors."""
    for attempt in Retrying(
        reraise=True,
        retry=retry_if_exception(_is_rate_limit_error),
        wait=wait_exponential(
            multiplier=config.RETRY_MULTIPLIER_SECONDS,
            min=config.RETRY_MIN_WAIT_SECONDS,
            max=config.RETRY_MAX_WAIT_SECONDS,
        ),
        stop=stop_after_attempt(config.RETRY_MAX_ATTEMPTS),
    ):
        with attempt:
            return llm.invoke(prompt)


def invoke_with_rate_limit(
    llm: BaseChatModel,
    prompt: str,
    limiter: _SlidingWindowRateLimiter,
) -> str:
    """Reserve an estimated token budget, invoke the model, and reconcile reported usage."""
    tokens = estimate_llm_tokens(prompt)
    limiter.acquire(tokens)

    response = _do_invoke_with_retry(llm, prompt)

    if hasattr(response, "response_metadata"):
        token_usage = response.response_metadata.get("token_usage", {})
        actual_tokens = token_usage.get("total_tokens")
        if actual_tokens is not None:
            limiter.adjust_tokens(tokens, actual_tokens)

    return _extract_content_str(response.content)


@locked_singleton
def get_input_guard_llm() -> ChatGroq:
    return _build_groq_llm(
        _require_groq_api_key(),
        config.INPUT_GUARD_MODEL,
        0.0,
    )


@locked_singleton
def get_output_guard_llm() -> ChatGroq:
    return _build_groq_llm(
        _require_groq_api_key(),
        config.OUTPUT_GUARD_MODEL,
        0.0,
    )


@locked_singleton
def get_router_llm() -> ChatGroq:
    return _build_groq_llm(
        _require_groq_api_key(),
        config.ROUTER_MODEL,
        config.ROUTING_LLM_TEMPERATURE,
        reasoning_effort="low",
        reasoning_format="hidden",
    )


@locked_singleton
def get_grader_llm() -> ChatGroq:
    return _build_groq_llm(
        _require_groq_api_key(),
        config.GRADER_MODEL,
        config.GRADER_LLM_TEMPERATURE,
    )


@locked_singleton
def get_rewriter_llm() -> ChatGroq:
    return _build_groq_llm(
        _require_groq_api_key(),
        config.REWRITER_MODEL,
        config.REWRITER_LLM_TEMPERATURE,
    )


@locked_singleton
def get_generation_llm() -> BaseChatModel:
    return _build_gemini_llm(
        _require_gemini_api_key(),
        config.GENERATION_MODEL,
        config.GENERATION_LLM_TEMPERATURE,
    )
