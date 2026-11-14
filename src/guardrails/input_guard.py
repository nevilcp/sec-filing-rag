"""Input guardrails: PII detection and a prompt injection check."""

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
    results = analyzer().analyze(text=text, language="en", entities=PII_ENTITIES)
    ordered_types: list[str] = []
    for item in results:
        entity_type = str(item.entity_type)
        if entity_type not in ordered_types:
            ordered_types.append(entity_type)
    return ordered_types


def _is_prompt_injection(query: str) -> bool:
    """Check the injection score, accepting unsafe or malicious verdicts as a fallback."""
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
        return "unsafe" in result or "malicious" in result


def validate_input(query: str) -> tuple[bool, str]:
    """Reject PII locally before checking prompt injection, returning validity and a reason."""
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
