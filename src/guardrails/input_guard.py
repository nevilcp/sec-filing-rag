"""Input guardrails: PII detection and a prompt injection check.

Responsibilities:
  - Block queries that contain PII, before any text leaves the process.
  - Block prompt injection and jailbreak attempts, using the Llama
    PromptGuard-2-86M classifier on Groq.
  - Open a trace span and write an audit-friendly log line per decision.

Dependencies:
  - GROQ_API_KEY, for the injection classifier.
  - The spaCy model en_core_web_lg, loaded through guardrails/pii.py.
"""

from __future__ import annotations

from opentelemetry.trace import Status, StatusCode

from src.core import config
from src.core.llm_factory import (
    get_input_guard_llm,
    invoke_with_rate_limit,
)
from src.core.logging import get_logger
from src.core.rate_limiter import get_input_guard_limiter
from src.core.tracing import chain_span, set_span_output
from src.guardrails.pii import PII_ENTITIES, analyzer

logger = get_logger(__name__)


def _detect_pii_entities(text: str) -> list[str]:
    """Return the distinct PII entity types found in the text.

    Args:
        text: The input text to scan for PII.

    Returns:
        Unique entity type names in detection order, for example
        ['EMAIL_ADDRESS', 'US_SSN'].
    """
    results = analyzer().analyze(text=text, language="en", entities=PII_ENTITIES)
    ordered_types: list[str] = []
    for item in results:
        entity_type = str(item.entity_type)
        if entity_type not in ordered_types:
            ordered_types.append(entity_type)
    return ordered_types


def _is_prompt_injection(query: str) -> bool:
    """Classify the input with PromptGuard-2-86M on Groq.

    The model replies with a score. If the reply is not a number, a
    textual verdict containing 'unsafe' or 'malicious' also counts.

    Args:
        query: Raw user input text.

    Returns:
        True if the injection score reaches
        config.INPUT_GUARD_INJECTION_THRESHOLD.

    Raises:
        RuntimeError: If GROQ_API_KEY is not set.
        RateLimitExhausted: If the query can never fit the model's
            token limit.
    """
    result = (
        invoke_with_rate_limit(
            get_input_guard_llm(),
            query,
            get_input_guard_limiter(),
        )
        .strip()
        .lower()
    )

    try:
        return float(result) >= config.INPUT_GUARD_INJECTION_THRESHOLD
    except ValueError:
        # Not a score: accept a textual verdict from another classifier.
        return "unsafe" in result or "malicious" in result


def validate_input(query: str) -> tuple[bool, str]:
    """Check a query for PII and prompt injection.

    PII is checked first, so a query that contains PII is rejected
    without being sent to the Groq classifier.

    Args:
        query: Raw user input text.

    Returns:
        A tuple of (is_valid, rejection_reason). The reason is an empty
        string when the query is valid.

    Raises:
        RuntimeError: If GROQ_API_KEY is not set.
        Exception: Errors from the Groq call, such as an API status
            error, propagate to the caller.

    Example:
        >>> validate_input("What were Apple's net sales in 2025?")
        (True, '')
        >>> validate_input("Reach me at jo@example.com")
        (False, "PII detected: ['EMAIL_ADDRESS']")
    """
    with chain_span("validate_input", query) as span:
        detected = _detect_pii_entities(query)
        if detected:
            reason = f"PII detected: {detected}"
            logger.warning("Input blocked due to PII", extra={"extra": {"entities": detected}})
            set_span_output(span, reason)
            span.set_status(Status(StatusCode.OK))
            return (False, reason)

        if _is_prompt_injection(query):
            reason = "Prompt injection detected"
            logger.warning("Input blocked due to prompt injection")
            set_span_output(span, reason)
            span.set_status(Status(StatusCode.OK))
            return (False, reason)

        logger.info("Input validation passed")
        set_span_output(span, "valid")
        span.set_status(Status(StatusCode.OK))
        return (True, "")
