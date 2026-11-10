"""
agent/state.py

State contract for the compliance LangGraph workflow.

Responsibilities:
  - Define the structured state payload passed between graph nodes
  - Enforce typing for retrieval results, CRAG tracking, and outputs

Dependencies:
  - typing
"""

from __future__ import annotations

from typing import Literal, TypedDict


class ComplianceState(TypedDict):
    """Immutable state contract passed between every node in the CRAG graph.

    All fields are set by specific nodes; downstream nodes must treat
    absent keys as their zero value (empty string, empty list, False).
    The graph compiler enforces this via LangGraph's TypedDict reducer.
    """

    query: str
    # The user's text as submitted. `query` is replaced by CRAG rewrites.
    original_query: str
    retrieved_docs: list[dict]
    retrieval_grade: Literal["relevant", "irrelevant"]
    rewrite_count: int
    final_response: str
    cache_hit: bool
    query_category: Literal[
        "compliance_check",
        "document_analysis",
        "cross_regulation",
        "cross_company",
        "temporal",
        "general",
    ]
    context_relevance: Literal["relevant", "irrelevant"] | None
    company_filter: str | None
