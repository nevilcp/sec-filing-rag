"""
core/llm_factory.py

Centralized LLM factory functions with cached instances and retry policy.

Responsibilities:
  - Initialize and cache LangChain Chat Models per pipeline node role
  - Inject rate-limiting barriers before executing model invocations
  - Execute API calls with exponential backoff for transient 429 errors

Dependencies:
  - langchain_core
  - langchain_groq
  - langchain_google_genai
  - tenacity
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
from src.core.rate_limiter import (
    _SlidingWindowRateLimiter,
    estimate_llm_tokens,
    get_generation_limiter,
    get_input_guard_limiter,
    get_output_guard_limiter,
    get_router_limiter,
    get_workhorse_limiter,
)

_INPUT_GUARD_LLM: BaseChatModel | None = None
_OUTPUT_GUARD_LLM: BaseChatModel | None = None
_ROUTER_LLM: BaseChatModel | None = None
_GRADER_LLM: BaseChatModel | None = None
_REWRITER_LLM: BaseChatModel | None = None
_GENERATION_LLM: BaseChatModel | None = None

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
    recovery window is hours — far too long for tenacity to wait inline.

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
    """Read and validate the GROQ_API_KEY environment variable.

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
    """Read and validate the GEMINI_API_KEY environment variable.

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
    """Construct a ChatGroq instance with project-standard retry settings.

    Args:
        api_key: Groq API key.
        model_name: Groq model identifier string.
        temperature: Sampling temperature for the model.
        reasoning_effort: Optional Groq reasoning effort level
            (e.g., 'none' to disable chain-of-thought).
        reasoning_format: Optional output format for reasoning
            (e.g., 'hidden' to suppress scratchpad in output).

    Returns:
        A configured ChatGroq instance.
    """
    kwargs: dict = {
        "model": model_name,
        "api_key": api_key,
        "temperature": temperature,
        "max_retries": config.RETRY_MAX_ATTEMPTS,
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
    """Construct a ChatGoogleGenerativeAI instance with standard retry settings.

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
        max_retries=config.RETRY_MAX_ATTEMPTS,
    )


def _extract_content_str(content: object) -> str:
    """Normalize LangChain response content to a plain string.

    Handles both simple string responses and multi-block list
    responses (produced by some vision/multimodal models). Falls back
    to str() coercion so unexpected types never crash the pipeline.

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
    """Invoke an LLM with exponential-backoff retry on rate-limit errors.

    Uses tenacity's Retrying context manager so each attempt can be
    instrumented independently. Only retries on transient per-minute
    throttle errors; daily quota exhaustion propagates immediately.

    Args:
        llm: The LangChain chat model to invoke.
        prompt: The prompt string to send.

    Returns:
        The raw LangChain response object.

    Raises:
        Exception: Re-raises the last exception after all retries
            are exhausted.
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

    Acquires a token budget from the limiter before making the API call,
    then adjusts the limiter's sliding window with actual token usage
    reported by the API to keep the estimate accurate over time.

    Args:
        llm: The LangChain chat model to invoke.
        prompt: The prompt string to send.
        limiter: The per-model rate limiter to acquire from.

    Returns:
        The response content as a plain string.
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


def get_input_guard_llm() -> ChatGroq:
    """Return the singleton PromptGuard-2-86M injection/jailbreak classifier.

    Initializes the model lazily on first call. Temperature is forced to
    0.0 for fully deterministic classification output.

    Returns:
        The cached ChatGroq instance for the InputGuard node.
    """
    global _INPUT_GUARD_LLM
    if _INPUT_GUARD_LLM is None:
        _INPUT_GUARD_LLM = _build_groq_llm(
            _require_groq_api_key(),
            config.INPUT_GUARD_MODEL,
            0.0,
        )
    return _INPUT_GUARD_LLM


def get_input_guard_limiter_fn() -> _SlidingWindowRateLimiter:
    """Return the singleton rate limiter for the InputGuard model."""
    return get_input_guard_limiter()


def get_output_guard_llm() -> ChatGroq:
    """Return the singleton GPT-OSS-Safeguard-20B output policy classifier.

    Temperature is forced to 0.0 for deterministic policy decisions.

    Returns:
        The cached ChatGroq instance for the OutputGuard node.
    """
    global _OUTPUT_GUARD_LLM
    if _OUTPUT_GUARD_LLM is None:
        _OUTPUT_GUARD_LLM = _build_groq_llm(
            _require_groq_api_key(),
            config.OUTPUT_GUARD_MODEL,
            0.0,
        )
    return _OUTPUT_GUARD_LLM


def get_output_guard_limiter_fn() -> _SlidingWindowRateLimiter:
    """Return the singleton rate limiter for the OutputGuard model."""
    return get_output_guard_limiter()


def get_router_llm() -> ChatGroq:
    """Return the singleton router LLM (config.ROUTER_MODEL).

    gpt-oss models cannot disable reasoning, so effort is set to 'low'
    to keep latency down, and format 'hidden' keeps chain-of-thought
    out of the JSON classification output the parser reads.

    Returns:
        The cached ChatGroq instance for the Router node.
    """
    global _ROUTER_LLM
    if _ROUTER_LLM is None:
        _ROUTER_LLM = _build_groq_llm(
            _require_groq_api_key(),
            config.ROUTER_MODEL,
            config.ROUTING_LLM_TEMPERATURE,
            reasoning_effort="low",
            reasoning_format="hidden",
        )
    return _ROUTER_LLM


def get_router_limiter_fn() -> _SlidingWindowRateLimiter:
    """Return the singleton rate limiter for the Router model."""
    return get_router_limiter()


def get_grader_llm() -> ChatGroq:
    """Return the singleton relevance grader LLM (config.GRADER_MODEL).

    Returns:
        The cached ChatGroq instance for the Grader node.
    """
    global _GRADER_LLM
    if _GRADER_LLM is None:
        _GRADER_LLM = _build_groq_llm(
            _require_groq_api_key(),
            config.GRADER_MODEL,
            config.GRADER_LLM_TEMPERATURE,
        )
    return _GRADER_LLM


def get_grader_limiter_fn() -> _SlidingWindowRateLimiter:
    """Return the shared rate limiter for the Grader/Rewriter model pair."""
    return get_workhorse_limiter()


def get_rewriter_llm() -> ChatGroq:
    """Return the singleton CRAG rewriter LLM (config.REWRITER_MODEL).

    Returns:
        The cached ChatGroq instance for the Rewriter node.
    """
    global _REWRITER_LLM
    if _REWRITER_LLM is None:
        _REWRITER_LLM = _build_groq_llm(
            _require_groq_api_key(),
            config.REWRITER_MODEL,
            config.REWRITER_LLM_TEMPERATURE,
        )
    return _REWRITER_LLM


def get_rewriter_limiter_fn() -> _SlidingWindowRateLimiter:
    """Return the shared rate limiter for the Rewriter/Grader model pair."""
    return get_workhorse_limiter()


def get_generation_llm() -> BaseChatModel:
    """Return the singleton Gemini generation LLM for final answer synthesis.

    Returns:
        The cached BaseChatModel instance for the Generation node.
    """
    global _GENERATION_LLM
    if _GENERATION_LLM is None:
        _GENERATION_LLM = _build_gemini_llm(
            _require_gemini_api_key(),
            config.GENERATION_MODEL,
            config.GENERATION_LLM_TEMPERATURE,
        )
    return _GENERATION_LLM


def get_generation_limiter_fn() -> _SlidingWindowRateLimiter:
    """Return the singleton rate limiter for the Generation model."""
    return get_generation_limiter()
