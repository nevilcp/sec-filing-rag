"""
Shared Presidio engines for the input and output guards.

Responsibilities:
  - Own the single AnalyzerEngine and AnonymizerEngine instances, so the
    spaCy model loads once instead of once per guard
  - Define the PII entity types both guards scan for

Dependencies:
  - spaCy model en_core_web_lg (downloaded separately)
"""

from __future__ import annotations

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

from src.core.lazy import locked_singleton

PII_ENTITIES = ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD", "US_SSN"]


@locked_singleton
def analyzer() -> AnalyzerEngine:
    """Return the shared Presidio AnalyzerEngine, building it on first call."""
    return AnalyzerEngine()


@locked_singleton
def anonymizer() -> AnonymizerEngine:
    """Return the shared Presidio AnonymizerEngine, building it on first call."""
    return AnonymizerEngine()
