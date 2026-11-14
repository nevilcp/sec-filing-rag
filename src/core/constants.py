"""Static lookup tables for the SEC compliance RAG system."""

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


# Shared by corpus ingestion and query ticker detection.
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
