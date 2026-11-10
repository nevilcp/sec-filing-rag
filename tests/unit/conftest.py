"""Pytest configuration for the unit test suite.

Responsibilities:
  - Pin environment variables before any src module is imported, so
    tests never read a developer's .env or reach a live API.

Dependencies:
  - python-dotenv does not override variables that are already set, so
    the defaults below win over any local .env file.
"""

import os

os.environ.setdefault("EDGAR_FISCAL_YEAR", "2024")
os.environ.setdefault("GROQ_API_KEY", "test")
os.environ.setdefault("GEMINI_API_KEY", "test")
os.environ.setdefault("PHOENIX_ENABLED", "")
