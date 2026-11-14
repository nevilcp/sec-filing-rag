"""Per-model sliding-window rate limiter for LLM endpoints."""

from __future__ import annotations

import threading
import time
from collections import deque

from src.core import config
from src.core.logging import get_logger

logger = get_logger(__name__)
_CHARS_PER_TOKEN: int = 4  # Approximation for English prose.

# Scales the prompt estimate to also cover completion tokens.
_LLM_TOKEN_MULTIPLIER: float = 1.5


def estimate_llm_tokens(prompt: str) -> int:
    """Estimate prompt and completion tokens using a character heuristic."""
    input_tokens = max(1, len(prompt) // _CHARS_PER_TOKEN)
    return int(input_tokens * _LLM_TOKEN_MULTIPLIER)


class RateLimitExhausted(Exception):
    """A request whose estimated token cost exceeds the entire per-minute budget."""

    pass


# Per model: requests/minute, requests/day, tokens/minute, tokens/day.
_MODEL_LIMITS: dict[str, tuple[int, int, int, int]] = {
    "meta-llama/llama-prompt-guard-2-86m": (30, 14_400, 15_000, 500_000),
    "openai/gpt-oss-safeguard-20b": (30, 1_000, config.OUTPUT_GUARD_TPM, 200_000),
    "openai/gpt-oss-20b": (30, 1_000, 8_000, 200_000),
    "openai/gpt-oss-120b": (30, 1_000, 8_000, 200_000),
    "gemini-3.5-flash": (15, 1_500, 1_000_000, 5_000_000),
    "gemini-3-flash-preview": (15, 1_500, 1_000_000, 5_000_000),
}


def get_model_limits(model_id: str) -> tuple[int, int, int, int]:
    if model_id not in _MODEL_LIMITS:
        raise ValueError(f"No rate-limit specification for model '{model_id}'. Known models: {sorted(_MODEL_LIMITS)}")
    return _MODEL_LIMITS[model_id]


class _SlidingWindowRateLimiter:
    """Thread-safe per-model request and token limits over a 60-second window."""

    _WINDOW_SECONDS: float = 60.0
    # Seconds to sleep between checks while waiting for capacity.
    _POLL_INTERVAL: float = 1.0

    def __init__(
        self,
        rpm: int,
        tpm: int,
        model_label: str,
    ) -> None:
        self._rpm = rpm
        self._tpm = tpm
        self._label = model_label

        self._min_interval: float = 60.0 / rpm

        self._request_log: deque[tuple[float, int]] = deque()
        self._last_request_time: float = 0.0
        self._lock = threading.Lock()

    def _purge_expired(self, now: float) -> None:
        cutoff = now - self._WINDOW_SECONDS
        while self._request_log and self._request_log[0][0] < cutoff:
            self._request_log.popleft()

    def _current_rpm(self) -> int:
        return len(self._request_log)

    def _current_tpm(self) -> int:
        return sum(tokens for _, tokens in self._request_log)

    def acquire(self, estimated_tokens: int) -> None:
        """Wait for request, token, and spacing capacity; reject requests that can never fit."""
        if estimated_tokens > self._tpm:
            raise RateLimitExhausted(
                f"[{self._label}] Request estimate {estimated_tokens} exceeds TPM limit {self._tpm}"
            )
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
                            "[%s] Resumed - RPM %d/%d  TPM %d/%d",
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
                            "[%s] Throttling - %s. Waiting...",
                            self._label,
                            ", ".join(reasons),
                        )
                    else:
                        logger.debug(
                            "[%s] Pacing request - %s. Waiting...",
                            self._label,
                            ", ".join(reasons),
                        )
                    waited = True

            time.sleep(self._POLL_INTERVAL)

    def stats(self) -> dict[str, int]:
        """Report current window usage, with daily counters fixed at zero."""
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
        """Replace the latest reservation with reported usage; the estimated argument is unused."""
        with self._lock:
            if self._request_log:
                timestamp, _ = self._request_log[-1]
                self._request_log[-1] = (timestamp, max(0, actual))


_LIMITER_CACHE: dict[str, _SlidingWindowRateLimiter] = {}
_LIMITER_LOCK = threading.Lock()


def _get_or_create_limiter(
    model_id: str,
) -> _SlidingWindowRateLimiter:
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
    return _get_or_create_limiter(config.INPUT_GUARD_MODEL)


def get_output_guard_limiter() -> _SlidingWindowRateLimiter:
    return _get_or_create_limiter(config.OUTPUT_GUARD_MODEL)


def get_router_limiter() -> _SlidingWindowRateLimiter:
    return _get_or_create_limiter(config.ROUTER_MODEL)


def get_workhorse_limiter() -> _SlidingWindowRateLimiter:
    """Return the shared per-model budget used by the grader and rewriter."""
    return _get_or_create_limiter(config.GRADER_MODEL)


def get_generation_limiter() -> _SlidingWindowRateLimiter:
    return _get_or_create_limiter(config.GENERATION_MODEL)


def get_embedding_rate_limiter() -> _SlidingWindowRateLimiter:
    """Return the workhorse limiter reported by the metrics endpoint."""
    return get_workhorse_limiter()
