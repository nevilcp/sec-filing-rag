REGULATION_NAME_MAP = {
    "10b-5": "Rule 10b-5",
    "10-k": "Form 10-K",
    "10-q": "Form 10-Q",
    "8-k": "Form 8-K",
    "13d": "Schedule 13D",
    "13g": "Schedule 13G",
    "144": "Rule 144",
    "fd": "Regulation FD",
    "reg fd": "Regulation FD",
    "s-k": "Regulation S-K",
    "s-x": "Regulation S-X",
    "reg sho": "Regulation SHO",
    "best interest": "Regulation Best Interest",
}


KNOWN_REGULATIONS = [
    "Rule 10b-5",
    "Form 10-K",
    "Form 10-Q",
    "Form 8-K",
    "Schedule 13D",
    "Schedule 13G",
    "Rule 144",
    "Regulation FD",
    "Regulation S-K",
    "Regulation S-X",
    "Regulation SHO",
    "Regulation Best Interest",
]


COMPANY_MANIFEST = [
    {"ticker": "AAPL", "name": "Apple Inc.", "sector": "technology"},
    {"ticker": "MSFT", "name": "Microsoft Corporation", "sector": "technology"},
    {"ticker": "GOOGL", "name": "Alphabet Inc.", "sector": "technology"},
    {"ticker": "NVDA", "name": "NVIDIA Corporation", "sector": "technology"},
    {"ticker": "JPM", "name": "JPMorgan Chase & Co.", "sector": "financials"},
    {"ticker": "BAC", "name": "Bank of America Corporation", "sector": "financials"},
    {"ticker": "GS", "name": "The Goldman Sachs Group, Inc.", "sector": "financials"},
    {"ticker": "MS", "name": "Morgan Stanley", "sector": "financials"},
    {"ticker": "JNJ", "name": "Johnson & Johnson", "sector": "healthcare"},
    {"ticker": "UNH", "name": "UnitedHealth Group Incorporated", "sector": "healthcare"},
    {"ticker": "PFE", "name": "Pfizer Inc.", "sector": "healthcare"},
    {"ticker": "ABT", "name": "Abbott Laboratories", "sector": "healthcare"},
]

KNOWN_TICKERS = [company["ticker"] for company in COMPANY_MANIFEST]

TICKER_TO_NAME = {company["ticker"]: company["name"] for company in COMPANY_MANIFEST}
