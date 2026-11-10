"""Centralized LLM factories with cached instances and a retry policy.

Responsibilities:
  - Build and cache one LangChain chat model per pipeline node role.
  - Acquire a rate-limit budget before every model invocation.
  - Retry transient per-minute 429 errors with exponential backoff.

Dependencies:
  - GROQ_API_KEY and GEMINI_API_KEY, read when a client is first built.
"""

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
    """Return True only for transient per-minute rate-limit errors.

    Deliberately excludes daily quota exhaustion (TPD/RPD) because the
    recovery window is hours, far too long for tenacity to wait inline.

    Args:
        exc: The exception raised by the LLM API call.

    Returns:
        True if the error is a retryable per-minute throttle.
    """
    message = str(exc).lower()

    if any(marker in message for marker in ("tokens per day", "tpd", "requests per day", "rpd")):
        return False

    return any(marker in message for marker in _RATE_LIMIT_ERROR_MARKERS)


def _require_groq_api_key() -> str:
    """Read the GROQ_API_KEY environment variable.

    Returns:
        The non-empty API key string.

    Raises:
        RuntimeError: If GROQ_API_KEY is absent or empty.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is required but not set. Define GROQ_API_KEY in your environment before creating LLM clients."
        )
    return api_key


def _require_gemini_api_key() -> str:
    """Read the GEMINI_API_KEY environment variable.

    Returns:
        The non-empty API key string.

    Raises:
        RuntimeError: If GEMINI_API_KEY is absent or empty.
    """
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
    """Build a ChatGroq client with the project retry settings.

    Args:
        api_key: Groq API key.
        model_name: Groq model identifier string.
        temperature: Sampling temperature for the model.
        reasoning_effort: Optional Groq reasoning effort level, for
            example 'low'.
        reasoning_format: Optional reasoning output format, for
            example 'hidden' to suppress the scratchpad.

    Returns:
        A configured ChatGroq instance.
    """
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
    """Build a ChatGoogleGenerativeAI client with the retry settings.

    Args:
        api_key: Google Gemini API key.
        model_name: Gemini model identifier string.
        temperature: Sampling temperature for the model.

    Returns:
        A configured ChatGoogleGenerativeAI instance.
    """
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=temperature,
        max_retries=config.LLM_SDK_MAX_RETRIES,
    )


def _extract_content_str(content: object) -> str:
    """Normalize LangChain response content to a plain string.

    Handles plain strings and multi-block lists, which some multimodal
    models return. Other types fall back to str() so an unexpected
    shape never crashes the pipeline.

    Args:
        content: The raw content field from a LangChain response.

    Returns:
        A single concatenated string of all text content.
    """
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
    """Invoke an LLM, retrying rate-limit errors with backoff.

    Only transient per-minute throttle errors are retried. Daily quota
    exhaustion propagates immediately.

    Args:
        llm: The LangChain chat model to invoke.
        prompt: The prompt string to send.

    Returns:
        The raw LangChain response object.

    Raises:
        Exception: The last error from the model call, once retries are
            exhausted or if the error is not a per-minute throttle.
    """
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
    """Invoke an LLM through the sliding-window rate limiter.

    Acquires a token budget from the limiter before the API call, then
    replaces the estimate with the actual token usage the API reports.

    Args:
        llm: The LangChain chat model to invoke.
        prompt: The prompt string to send.
        limiter: The per-model rate limiter to acquire from.

    Returns:
        The response content as a plain string.

    Raises:
        RateLimitExhausted: If the prompt's token estimate alone exceeds
            the limiter's per-minute token limit.
        Exception: The last error from the model call after retries (see
            _do_invoke_with_retry).

    Example:
        >>> llm = get_router_llm()
        >>> limiter = get_router_limiter()
        >>> text = invoke_with_rate_limit(llm, "Classify.", limiter)
    """
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
    """Return the singleton PromptGuard-2-86M injection classifier.

    Temperature is forced to 0.0 for deterministic classification.

    Returns:
        The cached ChatGroq instance for the InputGuard node.

    Raises:
        RuntimeError: If GROQ_API_KEY is not set.
    """
    return _build_groq_llm(
        _require_groq_api_key(),
        config.INPUT_GUARD_MODEL,
        0.0,
    )


@locked_singleton
def get_output_guard_llm() -> ChatGroq:
    """Return the singleton GPT-OSS-Safeguard-20B policy classifier.

    Temperature is forced to 0.0 for deterministic policy decisions.

    Returns:
        The cached ChatGroq instance for the OutputGuard node.

    Raises:
        RuntimeError: If GROQ_API_KEY is not set.
    """
    return _build_groq_llm(
        _require_groq_api_key(),
        config.OUTPUT_GUARD_MODEL,
        0.0,
    )


@locked_singleton
def get_router_llm() -> ChatGroq:
    """Return the singleton router LLM (config.ROUTER_MODEL).

    gpt-oss models cannot disable reasoning, so effort is set to 'low'
    to keep latency down, and format 'hidden' keeps chain-of-thought
    out of the JSON classification output the parser reads.

    Returns:
        The cached ChatGroq instance for the Router node.

    Raises:
        RuntimeError: If GROQ_API_KEY is not set.
    """
    return _build_groq_llm(
        _require_groq_api_key(),
        config.ROUTER_MODEL,
        config.ROUTING_LLM_TEMPERATURE,
        reasoning_effort="low",
        reasoning_format="hidden",
    )


@locked_singleton
def get_grader_llm() -> ChatGroq:
    """Return the singleton relevance grader LLM (config.GRADER_MODEL).

    Returns:
        The cached ChatGroq instance for the Grader node.

    Raises:
        RuntimeError: If GROQ_API_KEY is not set.
    """
    return _build_groq_llm(
        _require_groq_api_key(),
        config.GRADER_MODEL,
        config.GRADER_LLM_TEMPERATURE,
    )


@locked_singleton
def get_rewriter_llm() -> ChatGroq:
    """Return the singleton CRAG rewriter LLM (config.REWRITER_MODEL).

    Returns:
        The cached ChatGroq instance for the Rewriter node.

    Raises:
        RuntimeError: If GROQ_API_KEY is not set.
    """
    return _build_groq_llm(
        _require_groq_api_key(),
        config.REWRITER_MODEL,
        config.REWRITER_LLM_TEMPERATURE,
    )


@locked_singleton
def get_generation_llm() -> BaseChatModel:
    """Return the singleton Gemini LLM that writes the final answer.

    Returns:
        The cached BaseChatModel instance for the Generation node.

    Raises:
        RuntimeError: If GEMINI_API_KEY is not set.
    """
    return _build_gemini_llm(
        _require_gemini_api_key(),
        config.GENERATION_MODEL,
        config.GENERATION_LLM_TEMPERATURE,
    )
