"""Output guardrails using Presidio + LLM-as-judge checks."""

from __future__ import annotations

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from src.core.llm_factory import get_classification_llm
from src.core.logging import get_logger

logger = get_logger(__name__)

_ANALYZER: AnalyzerEngine | None = None
_ANONYMIZER: AnonymizerEngine | None = None
_PII_ENTITIES = ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD", "US_SSN"]


def _analyzer() -> AnalyzerEngine:
    global _ANALYZER
    if _ANALYZER is None:
        _ANALYZER = AnalyzerEngine()
    return _ANALYZER


def _anonymizer() -> AnonymizerEngine:
    global _ANONYMIZER
    if _ANONYMIZER is None:
        _ANONYMIZER = AnonymizerEngine()
    return _ANONYMIZER


def _anonymize_pii(text: str) -> tuple[str, list[str]]:
    detections = _analyzer().analyze(text=text, language="en", entities=_PII_ENTITIES)
    if not detections:
        return text, []

    entity_types = sorted({str(item.entity_type) for item in detections})
    operators = {
        entity_type: OperatorConfig("replace", {"new_value": f"<{entity_type}>"})
        for entity_type in entity_types
    }
    anonymized = _anonymizer().anonymize(
        text=text,
        analyzer_results=detections,
        operators=operators,
    )
    return anonymized.text, entity_types


def _is_definitive_legal_advice(text: str) -> bool:
    prompt = (
        "You are a strict legal-safety classifier.\n"
        "Decide if the assistant response contains definitive legal advice or legal conclusions "
        "presented as certain outcomes, instead of informational compliance guidance.\n"
        "Return exactly one token: YES or NO.\n\n"
        f"Response:\n{text}"
    )
    decision = str(get_classification_llm().invoke(prompt).content).strip().upper()
    return decision.startswith("YES")


def validate_output(response: str) -> tuple[str, list[str]]:
    warnings: list[str] = []
    sanitized, detected_types = _anonymize_pii(response)
    if detected_types:
        warnings.append(f"PII anonymized: {detected_types}")
        logger.warning(
            "Output contained PII and was anonymized",
            extra={"extra": {"entities": detected_types}},
        )

    if _is_definitive_legal_advice(sanitized):
        warnings.append(
            "Potential definitive legal advice detected; response should be treated as informational only."
        )
        logger.warning("Output flagged for definitive legal advice tone")

    logger.info("Output validation complete", extra={"extra": {"warnings_count": len(warnings)}})
    return sanitized, warnings
