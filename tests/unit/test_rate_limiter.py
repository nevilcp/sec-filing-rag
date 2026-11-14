"""Unit tests for the thread-safe sliding window rate limiter."""

from unittest import mock

import pytest

from src.core import config
from src.core.rate_limiter import (
    _LIMITER_CACHE,
    RateLimitExhausted,
    _get_or_create_limiter,
    _SlidingWindowRateLimiter,
    estimate_llm_tokens,
    get_model_limits,
)


def test_estimate_llm_tokens():
    assert estimate_llm_tokens("12345678901234567890") == 7


def test_get_model_limits_known():
    limits = get_model_limits("openai/gpt-oss-120b")
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


def test__get_or_create_limiter_caching():
    _LIMITER_CACHE.clear()

    limiter1 = _get_or_create_limiter("openai/gpt-oss-120b")
    limiter2 = _get_or_create_limiter("openai/gpt-oss-120b")
    limiter3 = _get_or_create_limiter("gemini-3.5-flash")

    assert limiter1 is limiter2
    assert limiter1 is not limiter3


@pytest.mark.parametrize(
    "model",
    [
        config.INPUT_GUARD_MODEL,
        config.OUTPUT_GUARD_MODEL,
        config.ROUTER_MODEL,
        config.GRADER_MODEL,
        config.REWRITER_MODEL,
        config.GENERATION_MODEL,
    ],
)
def test_model_limits_configured_model_has_entry(model):
    assert len(get_model_limits(model)) == 4


def test_acquire_estimate_above_tpm_raises_without_sleeping():
    limiter = _SlidingWindowRateLimiter(rpm=60, tpm=1000, model_label="test")

    # Sleeping means the request would wait forever, so fail fast on it.
    with mock.patch("time.sleep", side_effect=AssertionError("acquire slept")):
        with pytest.raises(RateLimitExhausted):
            limiter.acquire(1001)

    assert limiter._current_tpm() == 0


def test_acquire_estimate_equal_to_tpm_is_allowed():
    limiter = _SlidingWindowRateLimiter(rpm=60, tpm=1000, model_label="test")

    limiter.acquire(1000)

    assert limiter._current_tpm() == 1000
