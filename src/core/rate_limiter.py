"""Sliding-window rate limiter for Gemini Embedding API and LLM tiers.

Enforces three independent ceilings (RPM / TPM / RPD) and blocks the calling
thread until the request can proceed within all three windows simultaneously.
Includes a mandatory inter-request cooldown derived from the RPM ceiling to
prevent bursting past the API's server-side window.

Three limiters are exposed as module-level singletons:

* :func:`get_embedding_rate_limiter` — Gemini Embedding 2  (RPM 100, TPM 30K, RPD 1K)
* :func:`get_tier1_limiter`          — Gemini 3.1 Flash Lite (RPM 15,  TPM 250K, RPD 500)
* :func:`get_tier2_limiter`          — Gemini 2.5 Flash       (RPM 5,   TPM 250K, RPD 20)

The Tier 2 limiter persists its daily counter to the ingest checkpoint file
so that API process restarts do not silently reset a near-exhausted budget.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from datetime import date
from pathlib import Path
from typing import Sequence

from src.core import config
from src.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------
# Gemini tokenizer is roughly 1 token per 4 characters for English prose.
# We use a conservative multiplier so we slightly over-count rather than
# under-count, keeping us safely within the TPM ceiling.
_CHARS_PER_TOKEN: int = 4

# LLM responses add output tokens on top of input tokens.  Since TPM limits
# are 250K for both LLM tiers (far above real usage), a simple 1.5× input
# multiplier is sufficient to account for completions without needing
# post-call correction.
_LLM_TOKEN_MULTIPLIER: float = 1.5


def estimate_tokens(texts: Sequence[str]) -> int:
    """Return a conservative token estimate for a batch of strings."""
    return max(1, sum(len(t) for t in texts) // _CHARS_PER_TOKEN)


def estimate_llm_tokens(prompt: str) -> int:
    """Return a conservative token estimate for an LLM prompt + completion."""
    input_tokens = max(1, len(prompt) // _CHARS_PER_TOKEN)
    return int(input_tokens * _LLM_TOKEN_MULTIPLIER)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class RateLimitExhausted(Exception):
    """Raised when the daily RPD ceiling has been reached.

    Callers should catch this, checkpoint their progress, and abort the
    current session gracefully.  The limit resets at the next calendar day.
    """


# ---------------------------------------------------------------------------
# Base rate limiter
# ---------------------------------------------------------------------------
class _SlidingWindowRateLimiter:
    """Thread-safe sliding-window rate limiter.

    Tracks three independent dimensions:
    * **RPM** – requests per minute  (sliding 60 s window)
    * **TPM** – tokens per minute    (sliding 60 s window)
    * **RPD** – requests per day     (calendar-day counter, resets at midnight)

    Additionally enforces a minimum inter-request interval derived from the
    RPM ceiling (``60.0 / rpm``) to prevent burst-firing past the API's
    server-side rate window.

    Call :meth:`acquire` **before** every API call.  It will block (sleep)
    until the call can proceed within all three limits.

    Subclasses may override :meth:`_load_daily_count` and
    :meth:`_persist_daily_count` to add cross-process persistence.
    """

    _WINDOW_SECONDS: float = 60.0
    _POLL_INTERVAL: float = 1.0

    def __init__(
        self,
        rpm: int,
        tpm: int,
        rpd: int,
        model_label: str,
    ) -> None:
        self._rpm = rpm
        self._tpm = tpm
        self._rpd = rpd
        self._label = model_label

        # min_interval is derived from RPM so misconfiguration is impossible.
        self._min_interval: float = 60.0 / rpm

        # Sliding window: each entry is (unix_timestamp, token_count).
        self._request_log: deque[tuple[float, int]] = deque()

        # Daily counter — subclasses may seed this from persistent storage.
        self._daily_count: int = self._load_daily_count()
        self._daily_date: date = date.today()
        self._last_request_time: float = 0.0

        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Persistence hooks (no-op in base class)
    # ------------------------------------------------------------------
    def _load_daily_count(self) -> int:
        """Load the persisted daily count.  Returns 0 if not overridden."""
        return 0

    def _persist_daily_count(self) -> None:
        """Persist the current daily count.  No-op if not overridden."""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _purge_expired(self, now: float) -> None:
        """Remove entries older than the 60 s sliding window."""
        cutoff = now - self._WINDOW_SECONDS
        while self._request_log and self._request_log[0][0] < cutoff:
            self._request_log.popleft()

    def _reset_daily_if_needed(self) -> None:
        """Reset the RPD counter on calendar-day rollover."""
        today = date.today()
        if today != self._daily_date:
            logger.info(
                "[%s] Daily reset: %s -> %s (previous count: %d/%d)",
                self._label,
                self._daily_date.isoformat(),
                today.isoformat(),
                self._daily_count,
                self._rpd,
            )
            self._daily_count = 0
            self._daily_date = today
            self._persist_daily_count()

    def _current_rpm(self) -> int:
        return len(self._request_log)

    def _current_tpm(self) -> int:
        return sum(tokens for _, tokens in self._request_log)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def acquire(self, estimated_tokens: int) -> None:
        """Block until the request fits within RPM, TPM, RPD, and cooldown.

        Parameters
        ----------
        estimated_tokens:
            Conservative token estimate for the upcoming request.  Use
            :func:`estimate_tokens` for embedding calls and
            :func:`estimate_llm_tokens` for LLM calls.

        Raises
        ------
        RateLimitExhausted
            If the daily RPD ceiling has been reached.  Unlike RPM/TPM
            exhaustion (which resolves by waiting), RPD exhaustion cannot
            be resolved within the current calendar day.
        """
        waited = False
        while True:
            with self._lock:
                now = time.monotonic()
                self._purge_expired(now)
                self._reset_daily_if_needed()

                rpm_ok = self._current_rpm() < self._rpm
                tpm_ok = (self._current_tpm() + estimated_tokens) <= self._tpm
                rpd_ok = self._daily_count < self._rpd
                cooldown_ok = (now - self._last_request_time) >= self._min_interval

                if rpm_ok and tpm_ok and rpd_ok and cooldown_ok:
                    self._request_log.append((now, estimated_tokens))
                    self._daily_count += 1
                    self._last_request_time = now
                    self._persist_daily_count()
                    if waited:
                        logger.info(
                            "[%s] Resumed — RPM %d/%d  TPM %d/%d  RPD %d/%d",
                            self._label,
                            self._current_rpm(), self._rpm,
                            self._current_tpm(), self._tpm,
                            self._daily_count, self._rpd,
                        )
                    return

                # --- Must wait ---
                if not rpd_ok:
                    logger.error(
                        "[%s] STATUS: Daily request limit (%d) exhausted.",
                        self._label,
                        self._rpd
                    )
                    # RPD exhaustion cannot be resolved by waiting.
                    raise RateLimitExhausted(
                        f"[{self._label}] Daily limit reached "
                        f"({self._daily_count}/{self._rpd}). "
                        "Resume tomorrow."
                    )

                if not waited:
                    reasons: list[str] = []
                    if not rpm_ok:
                        reasons.append(f"RPM {self._current_rpm()}/{self._rpm}")
                    if not tpm_ok:
                        reasons.append(
                            f"TPM {self._current_tpm() + estimated_tokens}/{self._tpm}"
                        )
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

            # Release lock while sleeping so other threads are not blocked.
            time.sleep(self._POLL_INTERVAL)

    def budget_remaining(self) -> int:
        """Return the number of RPD requests remaining today.

        Thread-safe.  Callers can use this to implement graceful degradation
        before the hard :class:`RateLimitExhausted` exception fires.
        """
        with self._lock:
            self._reset_daily_if_needed()
            return max(0, self._rpd - self._daily_count)

    @property
    def daily_count(self) -> int:
        """Return the number of requests made today (thread-safe read)."""
        with self._lock:
            self._reset_daily_if_needed()
            return self._daily_count

    def stats(self) -> dict[str, int]:
        """Return a snapshot of current usage across all dimensions."""
        with self._lock:
            self._purge_expired(time.monotonic())
            self._reset_daily_if_needed()
            return {
                "rpm_used": self._current_rpm(),
                "rpm_limit": self._rpm,
                "tpm_used": self._current_tpm(),
                "tpm_limit": self._tpm,
                "rpd_used": self._daily_count,
                "rpd_limit": self._rpd,
            }


# ---------------------------------------------------------------------------
# Embedding limiter (unchanged public contract)
# ---------------------------------------------------------------------------
class EmbeddingRateLimiter(_SlidingWindowRateLimiter):
    """Rate limiter for Gemini Embedding 2.

    RPM 100  |  TPM 30,000  |  RPD 1,000
    min_interval derived as 60.0 / 100 = 0.6 s

    Both ingestion (batch embedding) and inference (query embedding) share
    this singleton, so the daily counter reflects total cross-process usage
    for the current calendar day.
    """

    def __init__(self) -> None:
        super().__init__(
            rpm=config.EMBEDDING_RATE_LIMIT_REQUESTS_PER_MINUTE,
            tpm=config.EMBEDDING_TOKENS_PER_MINUTE,
            rpd=config.EMBEDDING_REQUESTS_PER_DAY,
            model_label="embedding",
        )


# ---------------------------------------------------------------------------
# LLM Tier 1 limiter  —  Gemini 3.1 Flash Lite
# ---------------------------------------------------------------------------
class Tier1RateLimiter(_SlidingWindowRateLimiter):
    """Rate limiter for Gemini 3.1 Flash Lite (Tier 1).

    RPM 15  |  TPM 250,000  |  RPD 500
    min_interval derived as 60.0 / 15 = 4.0 s

    Used for: query routing, retrieval grading, query rewriting,
    guardrail LLM-as-judge calls, and fallback answer generation
    when the Tier 2 budget is near exhaustion.
    """

    def __init__(self) -> None:
        super().__init__(
            rpm=config.TIER1_RATE_LIMIT_REQUESTS_PER_MINUTE,
            tpm=config.TIER1_TOKENS_PER_MINUTE,
            rpd=config.TIER1_REQUESTS_PER_DAY,
            model_label="tier1-flash-lite",
        )


# ---------------------------------------------------------------------------
# LLM Tier 2 limiter  —  Gemini 2.5 Flash  (with checkpoint persistence)
# ---------------------------------------------------------------------------
class Tier2RateLimiter(_SlidingWindowRateLimiter):
    """Rate limiter for Gemini 2.5 Flash (Tier 2).

    RPM 5  |  TPM 250,000  |  RPD 20
    min_interval derived as 60.0 / 5 = 12.0 s

    Used exclusively for final answer generation in the ``respond()`` node.

    **Persistence:** The daily counter is read from and written to the ingest
    checkpoint file so that API process restarts do not silently reset a
    near-exhausted budget.  At only 20 RPD, losing track of 18 consumed
    requests on restart would exhaust the day's budget almost immediately.

    **Graceful degradation:** Callers should consult :meth:`budget_remaining`
    before every call.  If the remaining budget is ≤ ``TIER2_DEGRADATION_BUFFER``
    (default 2), fall back to Tier 1 for answer generation rather than
    exhausting the last requests unintentionally.
    """

    # How many requests to keep in reserve as a buffer against concurrent
    # calls and checkpoint race conditions.
    DEGRADATION_BUFFER: int = 2

    def __init__(self) -> None:
        super().__init__(
            rpm=config.TIER2_RATE_LIMIT_REQUESTS_PER_MINUTE,
            tpm=config.TIER2_TOKENS_PER_MINUTE,
            rpd=config.TIER2_REQUESTS_PER_DAY,
            model_label="tier2-flash",
        )

    # ------------------------------------------------------------------
    # Persistence implementation
    # ------------------------------------------------------------------
    @property
    def _checkpoint_path(self) -> Path:
        return Path(config.INGEST_CHECKPOINT_PATH)

    def _load_daily_count(self) -> int:
        """Load today's Tier 2 RPD count from the checkpoint file.

        Returns 0 if the file does not exist, cannot be parsed, or was
        written on a different calendar day.
        """
        path = Path(config.INGEST_CHECKPOINT_PATH)
        if not path.exists():
            return 0
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            stored_date = data.get("tier2_rpd_date")
            stored_count = data.get("tier2_rpd_used", 0)
            if stored_date == date.today().isoformat():
                logger.info(
                    "[tier2-flash] Restored daily count %d/%d from checkpoint.",
                    stored_count,
                    config.TIER2_REQUESTS_PER_DAY,
                )
                return int(stored_count)
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            logger.warning(
                "[tier2-flash] Could not load daily count from checkpoint: %s", exc
            )
        return 0

    def _persist_daily_count(self) -> None:
        """Write the current daily count back to the checkpoint file atomically.

        Reads the existing checkpoint, updates only the tier2 fields, and
        writes the result back.  Uses a temporary file + rename for
        atomicity so a crash mid-write does not corrupt the checkpoint.
        """
        path = Path(config.INGEST_CHECKPOINT_PATH)
        try:
            data: dict = {}
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    # Corrupt checkpoint — start fresh but preserve tier2 fields.
                    data = {}

            data["tier2_rpd_date"] = self._daily_date.isoformat()
            data["tier2_rpd_used"] = self._daily_count

            # Atomic write via tmp file + rename.
            tmp_path = path.with_suffix(".tmp")
            tmp_path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            tmp_path.replace(path)

        except OSError as exc:
            # Non-fatal: a failed write means the count is only in memory
            # for this session.  Log a warning so operators are aware.
            logger.warning(
                "[tier2-flash] Could not persist daily count to checkpoint: %s", exc
            )

    def should_degrade(self) -> bool:
        """Return True if the caller should fall back to Tier 1.

        Degradation is triggered when the remaining budget is within
        ``DEGRADATION_BUFFER`` of exhaustion, reserving those requests
        as a safety margin against concurrent calls.
        """
        return self.budget_remaining() <= self.DEGRADATION_BUFFER


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------
_EMBEDDING_LIMITER: EmbeddingRateLimiter | None = None
_TIER1_LIMITER: Tier1RateLimiter | None = None
_TIER2_LIMITER: Tier2RateLimiter | None = None

_EMBEDDING_LOCK = threading.Lock()
_TIER1_LOCK = threading.Lock()
_TIER2_LOCK = threading.Lock()


def get_embedding_rate_limiter() -> EmbeddingRateLimiter:
    """Return (or create) the module-level embedding rate limiter singleton."""
    global _EMBEDDING_LIMITER
    if _EMBEDDING_LIMITER is None:
        with _EMBEDDING_LOCK:
            if _EMBEDDING_LIMITER is None:
                _EMBEDDING_LIMITER = EmbeddingRateLimiter()
    return _EMBEDDING_LIMITER


def get_tier1_limiter() -> Tier1RateLimiter:
    """Return (or create) the Tier 1 (Gemini 3.1 Flash Lite) rate limiter."""
    global _TIER1_LIMITER
    if _TIER1_LIMITER is None:
        with _TIER1_LOCK:
            if _TIER1_LIMITER is None:
                _TIER1_LIMITER = Tier1RateLimiter()
    return _TIER1_LIMITER


def get_tier2_limiter() -> Tier2RateLimiter:
    """Return (or create) the Tier 2 (Gemini 2.5 Flash) rate limiter."""
    global _TIER2_LIMITER
    if _TIER2_LIMITER is None:
        with _TIER2_LOCK:
            if _TIER2_LIMITER is None:
                _TIER2_LIMITER = Tier2RateLimiter()
    return _TIER2_LIMITER