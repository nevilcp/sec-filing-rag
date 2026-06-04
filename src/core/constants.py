"""
core/constants.py

Static lookup tables for the SEC compliance RAG system.

Responsibilities:
  - Define the set of recognizable ticker symbols for query-directed
    retrieval (KNOWN_TICKERS).
  - Define the set of recognizable SEC/CFR regulation names for directed
    retrieval (KNOWN_REGULATIONS).

Not responsible for:
  - CIK resolution (see core/corpus.py).
  - Retrieval filtering logic (see rag/retriever.py).

Dependencies:
  - None
"""

KNOWN_REGULATIONS = [
    "Rule 10b-5",
    "Rule 10b5-1",
    "Regulation S-K",
    "Regulation S-X",
    "Regulation FD",
    "SOX Section 302",
    "SOX Section 404",
    "Item 105",
    "Item 106",
    "Item 303",
    "Item 305",
]


KNOWN_TICKERS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "NVDA",
    "JPM",
    "BAC",
    "GS",
    "MS",
    "JNJ",
    "UNH",
    "PFE",
    "ABT",
]
