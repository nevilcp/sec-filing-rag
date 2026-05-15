"""Hardcoded evaluation dataset for SEC compliance RAG."""

from __future__ import annotations

EVAL_DATASET = [
    # --- Single-Company Disclosure ---
    {
        "question": "How does JPM characterize its exposure to credit risk from commercial real estate lending, and what mitigation strategies does it describe in its 10-K?",
        "expected_topics": ["loan loss reserves", "commercial real estate concentration", "credit risk management", "collateral requirements"],
        "should_cite_regulation": False,
        "regulation_name": None,
    },
    {
        "question": "What operational and regulatory risks does UNH identify related to its government-sponsored health plan contracts, including Medicare and Medicaid?",
        "expected_topics": ["reimbursement rate changes", "CMS contract renewal", "regulatory compliance", "government revenue dependency"],
        "should_cite_regulation": False,
        "regulation_name": None,
    },

    # --- Cross-Company Comparison ---
    {
        "question": "Compare how NVDA and MSFT describe the competitive risks posed by new market entrants and rapid technological change in their respective 10-K filings.",
        "expected_topics": ["product obsolescence", "R&D investment", "competitor innovation", "market share erosion"],
        "should_cite_regulation": False,
        "regulation_name": None,
    },
    {
        "question": "How do JPM and GS differ in the way they disclose exposure to interest rate volatility and its projected impact on net interest income?",
        "expected_topics": ["rate sensitivity", "net interest margin", "fixed vs variable exposure", "hedging strategy"],
        "should_cite_regulation": False,
        "regulation_name": None,
    },

    # --- Section-Specific Retrieval ---
    {
        "question": "What quantitative disclosures does JPM provide in Item 7A regarding market risk from trading assets and value-at-risk methodologies?",
        "expected_topics": ["value at risk", "trading portfolio sensitivity", "market risk models", "stress testing"],
        "should_cite_regulation": True,
        "regulation_name": "Form 10-K",
    },
    {
        "question": "What does NVDA's 10-K Item 1A specifically disclose about export control regulations and restrictions on sales to certain international markets?",
        "expected_topics": ["export license requirements", "China trade restrictions", "Entity List compliance", "revenue impact from controls"],
        "should_cite_regulation": True,
        "regulation_name": "Form 10-K",
    },

    # --- General Compliance Reasoning ---
    {
        "question": "Based on the retrieved filings, how do the major financial institutions in the corpus frame their exposure to evolving digital asset and cryptocurrency regulation?",
        "expected_topics": ["regulatory uncertainty", "client demand for digital assets", "compliance investment", "reputational risk"],
        "should_cite_regulation": False,
        "regulation_name": None,
    },
    {
        "question": "Across the healthcare companies in the corpus, how consistently do they disclose litigation risk arising from drug pricing scrutiny and government investigations?",
        "expected_topics": ["DOJ investigations", "pricing transparency pressure", "class action exposure", "legal reserve disclosures"],
        "should_cite_regulation": False,
        "regulation_name": None,
    },
]
