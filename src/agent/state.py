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

from typing import Any, Literal, TypedDict

QueryCategory = Literal[
    "compliance_check",
    "document_analysis",
    "cross_regulation",
    "cross_company",
    "temporal",
    "general",
]


class RetrievedDoc(TypedDict):
    """One retrieved filing chunk with its score and provenance."""

    text: str
    score: float
    metadata: dict[str, Any]


def _doc_meta(doc: RetrievedDoc) -> dict[str, Any]:
    """Return a document's metadata dict, or an empty dict if absent."""
    return doc.get("metadata", {})


class ComplianceState(TypedDict):
    """Immutable state contract passed between every node in the CRAG graph.

    All fields are set by specific nodes; downstream nodes must treat
    absent keys as their zero value (empty string, empty list, False).
    The graph compiler enforces this via LangGraph's TypedDict reducer.
    """

    query: str
    # The user's text as submitted. `query` is replaced by CRAG rewrites.
    original_query: str
    retrieved_docs: list[RetrievedDoc]
    retrieval_grade: Literal["relevant", "irrelevant"]
    rewrite_count: int
    final_response: str
    cache_hit: bool
    query_category: QueryCategory
    context_relevance: Literal["relevant", "irrelevant"] | None
    company_filter: str | None
