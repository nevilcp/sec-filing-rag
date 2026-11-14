"""Shared Presidio engines for the input and output guards."""

from __future__ import annotations

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

from src.core.lazy import locked_singleton

PII_ENTITIES = ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD", "US_SSN"]


@locked_singleton
def analyzer() -> AnalyzerEngine:
    return AnalyzerEngine()


@locked_singleton
def anonymizer() -> AnonymizerEngine:
    return AnonymizerEngine()
