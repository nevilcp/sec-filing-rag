"""Shared Presidio engines for the input and output guards."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.lazy import locked_singleton

if TYPE_CHECKING:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine

PII_ENTITIES = ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD", "US_SSN"]


@locked_singleton
def analyzer() -> AnalyzerEngine:
    from presidio_analyzer import AnalyzerEngine

    return AnalyzerEngine()


@locked_singleton
def anonymizer() -> AnonymizerEngine:
    from presidio_anonymizer import AnonymizerEngine

    return AnonymizerEngine()
