"""
tests/unit/test_rate_limiter.py

Unit tests for the thread-safe sliding window rate limiter.

Responsibilities:
  - Validate token estimation multipliers
  - Ensure request/token buckets enforce exact constraints
  - Test cooldown waiting logic and window purging behavior
"""

from unittest import mock

import pytest

from src.core.rate_limiter import (
    _LIMITER_CACHE,
    _SlidingWindowRateLimiter,
    estimate_llm_tokens,
    get_limiter_for_model,
    get_model_limits,
)


def test_estimate_llm_tokens():
    assert estimate_llm_tokens("12345678901234567890") == 7


def test_get_model_limits_known():
    limits = get_model_limits("llama-3.1-8b-instant")
    assert len(limits) == 4


def test_get_model_limits_unknown():
    with pytest.raises(ValueError):
        get_model_limits("nonexistent-model")


def test_sliding_window_acquire_under_limit():
    limiter = _SlidingWindowRateLimiter(rpm=60, tpm=1000, model_label="test")

    with (
        mock.patch("time.monotonic") as mock_time,
        mock.patch("time.sleep") as mock_sleep,
    ):
        mock_time.return_value = 100.0
        limiter.acquire(10)
        assert limiter._current_rpm() == 1
        assert limiter._current_tpm() == 10
        mock_sleep.assert_not_called()


def test_sliding_window_cooldown():
    limiter = _SlidingWindowRateLimiter(rpm=60, tpm=1000, model_label="test")

    with (
        mock.patch("time.monotonic") as mock_time,
        mock.patch("time.sleep") as mock_sleep,
    ):
        mock_time.return_value = 100.0
        limiter.acquire(10)

        mock_time.side_effect = [100.5, 101.1]
        limiter.acquire(10)

        mock_sleep.assert_called_once_with(1.0)
        assert limiter._current_rpm() == 2


def test_sliding_window_rpm_limit():
    limiter = _SlidingWindowRateLimiter(rpm=2, tpm=1000, model_label="test")

    with (
        mock.patch("time.monotonic") as mock_time,
        mock.patch("time.sleep") as mock_sleep,
    ):
        mock_time.return_value = 100.0
        limiter.acquire(10)

        mock_time.return_value = 131.0
        limiter.acquire(10)

        assert limiter._current_rpm() == 2

        mock_time.side_effect = [150.0, 150.0, 161.0]
        limiter.acquire(10)

        assert mock_sleep.call_count >= 1
        assert limiter._current_rpm() == 2


def test_sliding_window_tpm_limit():
    limiter = _SlidingWindowRateLimiter(rpm=60, tpm=100, model_label="test")

    with (
        mock.patch("time.monotonic") as mock_time,
        mock.patch("time.sleep") as mock_sleep,
    ):
        mock_time.return_value = 100.0
        limiter.acquire(60)

        mock_time.side_effect = [102.0, 102.0, 161.0]
        limiter.acquire(50)

        assert mock_sleep.call_count >= 1
        assert limiter._current_tpm() == 50


def test_sliding_window_purge():
    limiter = _SlidingWindowRateLimiter(rpm=60, tpm=1000, model_label="test")

    with mock.patch("time.monotonic") as mock_time:
        mock_time.return_value = 100.0
        limiter.acquire(10)

        mock_time.return_value = 161.0
        limiter._purge_expired(161.0)
        assert limiter._current_rpm() == 0


def test_adjust_tokens():
    limiter = _SlidingWindowRateLimiter(rpm=60, tpm=1000, model_label="test")

    with mock.patch("time.monotonic") as mock_time:
        mock_time.return_value = 100.0
        limiter.acquire(100)

        assert limiter._current_tpm() == 100
        limiter.adjust_tokens(100, 50)
        assert limiter._current_tpm() == 50


def test_get_limiter_for_model_caching():
    _LIMITER_CACHE.clear()

    limiter1 = get_limiter_for_model("llama-3.1-8b-instant")
    limiter2 = get_limiter_for_model("llama-3.1-8b-instant")
    limiter3 = get_limiter_for_model("gemini-3.5-flash")

    assert limiter1 is limiter2
    assert limiter1 is not limiter3
