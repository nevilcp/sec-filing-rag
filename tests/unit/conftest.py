"""Pytest configuration for the unit test suite."""

import os

os.environ.setdefault("EDGAR_FISCAL_YEAR", "2024")
os.environ.setdefault("GROQ_API_KEY", "test")
os.environ.setdefault("GEMINI_API_KEY", "test")
os.environ.setdefault("PHOENIX_ENABLED", "")
