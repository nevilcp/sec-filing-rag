"""Static lookup tables for the SEC compliance RAG system.

Responsibilities:
  - Define the ticker symbols in the corpus (KNOWN_TICKERS).
  - Define the SEC/CFR regulation names the retriever recognizes in a
    query (KNOWN_REGULATIONS).

Not responsible for:
  - CIK resolution (see core/corpus.py).
  - Retrieval filtering logic (see rag/retriever.py).
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


# Tickers to ingest. Read by core/corpus.py to build the corpus and by
# rag/retriever.py to detect a company named in a query.
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
