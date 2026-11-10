"""
guardrails/input_guard.py

Input guardrails using Presidio PII detection + Llama PromptGuard-2-86M classifier.

Responsibilities:
  - Screen incoming queries for Personally Identifiable Information (PII)
  - Detect prompt injection and jailbreak attempts using a dedicated classifier
  - Gate processing of malicious or sensitive inputs

Dependencies:
  - presidio_analyzer
  - langchain_groq
"""

from __future__ import annotations

from opentelemetry.trace import Status, StatusCode
from presidio_analyzer import AnalyzerEngine

from src.core import config
from src.core.llm_factory import (
    get_input_guard_limiter_fn,
    get_input_guard_llm,
    invoke_with_rate_limit,
)
from src.core.logging import get_logger
from src.core.tracing import chain_span, set_span_output

logger = get_logger(__name__)

_ANALYZER: AnalyzerEngine | None = None
_PII_ENTITIES = ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD", "US_SSN"]


def _analyzer() -> AnalyzerEngine:
    """Return the singleton Presidio AnalyzerEngine, initializing on first call."""
    global _ANALYZER
    if _ANALYZER is None:
        _ANALYZER = AnalyzerEngine()
    return _ANALYZER


def _detect_pii_entities(text: str) -> list[str]:
    """Return a deduplicated, ordered list of detected PII entity types.

    Args:
        text: The input text to scan for PII.

    Returns:
        A list of unique entity type strings in detection order
        (e.g. ['EMAIL_ADDRESS', 'US_SSN']).
    """
    results = _analyzer().analyze(text=text, language="en", entities=_PII_ENTITIES)
    ordered_types: list[str] = []
    for item in results:
        entity_type = str(item.entity_type)
        if entity_type not in ordered_types:
            ordered_types.append(entity_type)
    return ordered_types


def _is_prompt_injection(query: str) -> bool:
    """Classifies the input using PromptGuard-2-86M on Groq.

    Args:
        query: Raw user input text.

    Returns:
        True if the injection score reaches the configured threshold.
    """
    result = (
        invoke_with_rate_limit(
            get_input_guard_llm(),
            query,
            get_input_guard_limiter_fn(),
        )
        .strip()
        .lower()
    )

    try:
        return float(result) >= config.INPUT_GUARD_INJECTION_THRESHOLD
    except ValueError:
        # Not a score: accept a textual verdict from a different classifier.
        return "unsafe" in result or "malicious" in result


def validate_input(query: str) -> tuple[bool, str]:
    """Validates input against PII leakage and injection attacks.

    Args:
        query: Raw user input text.

    Returns:
        A tuple of (is_valid, rejection_reason). is_valid is True if the query
        passes all checks
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
