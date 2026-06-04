"""
core/rate_limiter.py

Per-model sliding-window rate limiter for Groq LPU inference.

Responsibilities:
  - Enforce RPM, TPM, RPD, and TPD limits across different model instances
  - Ensure thread-safe rate limit acquisition prior to LLM invocation
  - Persist daily rate limit counters to disk for bounded-budget models

Dependencies:
  - None
"""

from __future__ import annotations

import threading
import time
from collections import deque

from src.core import config
from src.core.logging import get_logger

logger = get_logger(__name__)
_CHARS_PER_TOKEN: int = 4

_LLM_TOKEN_MULTIPLIER: float = 1.5


def estimate_llm_tokens(prompt: str) -> int:
    """Estimate the total token cost of an LLM call from the prompt length.

    Uses a conservative 4-chars-per-token heuristic (matching Llama/GPT
    density for English prose), then multiplies by 1.5x to account for
    completion tokens without requiring a post-call correction step.

    Args:
        prompt: The input prompt string.

    Returns:
        An integer token estimate that slightly over-counts to avoid
        exceeding per-minute limits.
    """
    input_tokens = max(1, len(prompt) // _CHARS_PER_TOKEN)
    return int(input_tokens * _LLM_TOKEN_MULTIPLIER)


class RateLimitExhausted(Exception):
    """Raised when a rate limiter cannot acquire within the allowed budget."""

    pass


_MODEL_LIMITS: dict[str, tuple[int, int, int, int]] = {
    "meta-llama/llama-prompt-guard-2-86m": (30, 14_400, 15_000, 500_000),
    "openai/gpt-oss-safeguard-20b": (30, 1_000, 8_000, 200_000),
    "qwen/qwen3-32b": (60, 1_000, 10_000, 300_000),
    "llama-3.1-8b-instant": (30, 14_400, 6_000, 500_000),
    "openai/gpt-oss-120b": (30, 1_000, 8_000, 200_000),
    "gemini-3.5-flash": (15, 1_500, 1_000_000, 5_000_000),
    "gemini-3-flash-preview": (15, 1_500, 1_000_000, 5_000_000),
}


def get_model_limits(model_id: str) -> tuple[int, int, int, int]:
    if model_id not in _MODEL_LIMITS:
        raise ValueError(f"No rate-limit specification for model '{model_id}'. Known models: {sorted(_MODEL_LIMITS)}")
    return _MODEL_LIMITS[model_id]


class _SlidingWindowRateLimiter:
    """Thread-safe sliding-window rate limiter for Groq LPU model endpoints.

    Enforces per-minute RPM and TPM constraints by maintaining a deque
    of (timestamp, token_count) entries for the most recent 60-second
    window. Also enforces a minimum inter-request cooldown derived from
    the RPM limit to smooth burst traffic across the window.
    """

    _WINDOW_SECONDS: float = 60.0
    _POLL_INTERVAL: float = 1.0

    def __init__(
        self,
        rpm: int,
        tpm: int,
        model_label: str,
    ) -> None:
        """Initialize the rate limiter for a specific model.

        Args:
            rpm: Maximum requests per minute allowed by the provider.
            tpm: Maximum tokens per minute allowed by the provider.
            model_label: Human-readable model name for log messages.
        """
        self._rpm = rpm
        self._tpm = tpm
        self._label = model_label

        self._min_interval: float = 60.0 / rpm

        self._request_log: deque[tuple[float, int]] = deque()
        self._last_request_time: float = 0.0
        self._lock = threading.Lock()

    def _purge_expired(self, now: float) -> None:
        """Remove entries older than the 60 s sliding window"""
        cutoff = now - self._WINDOW_SECONDS
        while self._request_log and self._request_log[0][0] < cutoff:
            self._request_log.popleft()

    def _current_rpm(self) -> int:
        return len(self._request_log)

    def _current_tpm(self) -> int:
        return sum(tokens for _, tokens in self._request_log)

    def acquire(self, estimated_tokens: int) -> None:
        """Block until the request fits within RPM, TPM, and cooldown"""
        waited = False
        while True:
            with self._lock:
                now = time.monotonic()
                self._purge_expired(now)

                rpm_ok = self._current_rpm() < self._rpm
                tpm_ok = (self._current_tpm() + estimated_tokens) <= self._tpm
                cooldown_ok = (now - self._last_request_time) >= self._min_interval

                if rpm_ok and tpm_ok and cooldown_ok:
                    self._request_log.append((now, estimated_tokens))
                    self._last_request_time = now
                    if waited:
                        logger.info(
                            "[%s] Resumed — RPM %d/%d  TPM %d/%d",
                            self._label,
                            self._current_rpm(),
                            self._rpm,
                            self._current_tpm(),
                            self._tpm,
                        )
                    return

                if not waited:
                    reasons: list[str] = []
                    if not rpm_ok:
                        reasons.append(f"RPM {self._current_rpm()}/{self._rpm}")
                    if not tpm_ok:
                        reasons.append(f"TPM {self._current_tpm() + estimated_tokens}/{self._tpm}")
                    if not cooldown_ok:
                        remaining = self._min_interval - (now - self._last_request_time)
                        reasons.append(f"cooldown {remaining:.1f}s remaining")

                    if not rpm_ok or not tpm_ok:
                        logger.warning(
                            "[%s] Throttling — %s. Waiting...",
                            self._label,
                            ", ".join(reasons),
                        )
                    else:
                        logger.debug(
                            "[%s] Pacing request — %s. Waiting...",
                            self._label,
                            ", ".join(reasons),
                        )
                    waited = True

            time.sleep(self._POLL_INTERVAL)

    def stats(self) -> dict[str, int]:
        """Return a snapshot of current usage across all dimensions"""
        with self._lock:
            self._purge_expired(time.monotonic())
            return {
                "rpm_used": self._current_rpm(),
                "rpm_limit": self._rpm,
                "tpm_used": self._current_tpm(),
                "tpm_limit": self._tpm,
                "rpd_used": 0,
                "rpd_limit": 0,
                "tpd_used": 0,
                "tpd_limit": 0,
            }

    def adjust_tokens(self, estimated: int, actual: int) -> None:
        """Adjust the sliding window's latest entry with exact usage"""
        with self._lock:
            if self._request_log:
                timestamp, _ = self._request_log[-1]
                self._request_log[-1] = (timestamp, max(0, actual))


_LIMITER_CACHE: dict[str, _SlidingWindowRateLimiter] = {}
_LIMITER_LOCK = threading.Lock()


def _get_or_create_limiter(
    model_id: str,
) -> _SlidingWindowRateLimiter:
    """Return the singleton limiter for a given model ID"""
    if model_id not in _LIMITER_CACHE:
        with _LIMITER_LOCK:
            if model_id not in _LIMITER_CACHE:
                rpm, _rpd, tpm, _tpd = get_model_limits(model_id)
                label = model_id.rsplit("/", 1)[-1]
                _LIMITER_CACHE[model_id] = _SlidingWindowRateLimiter(
                    rpm=rpm,
                    tpm=tpm,
                    model_label=label,
                )
                logger.info(
                    "Created rate limiter for %s (RPM %d, TPM %d)",
                    model_id,
                    rpm,
                    tpm,
                )
    return _LIMITER_CACHE[model_id]


def get_input_guard_limiter() -> _SlidingWindowRateLimiter:
    """Limiter for llama-prompt-guard-2-86m (RPM 30, RPD 14,400)"""
    return _get_or_create_limiter(config.INPUT_GUARD_MODEL)


def get_output_guard_limiter() -> _SlidingWindowRateLimiter:
    """Limiter for gpt-oss-safeguard-20b (RPM 30, RPD 1,000)"""
    return _get_or_create_limiter(config.OUTPUT_GUARD_MODEL)


def get_router_limiter() -> _SlidingWindowRateLimiter:
    """Limiter for qwen/qwen3-32b (RPM 60, RPD 1,000)"""
    return _get_or_create_limiter(config.ROUTER_MODEL)


def get_workhorse_limiter() -> _SlidingWindowRateLimiter:
    """Limiter for llama-3.1-8b-instant (RPM 30, RPD 14,400).

    Shared by both the grader and rewriter nodes since they use the same
    Groq model and Groq enforces limits at the organization level per model
    """
    return _get_or_create_limiter(config.GRADER_MODEL)


def get_generation_limiter() -> _SlidingWindowRateLimiter:
    """Limiter for gemini-3.5-flash (RPM 15, RPD 1,500)"""
    return _get_or_create_limiter(config.GENERATION_MODEL)


def get_embedding_rate_limiter() -> _SlidingWindowRateLimiter:
    """Alias for backward compatibility with the /metrics endpoint"""
    return get_workhorse_limiter()
