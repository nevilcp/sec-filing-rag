"""State contract for the compliance LangGraph workflow.

Responsibilities:
  - Define the typed state dict that LangGraph passes between nodes.
  - Define the shared document and query-category types.

Not responsible for:
  - Runtime validation. The types are for static checking only.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

# The values the router can assign to ComplianceState.query_category.
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
    """State shared by every node in the CRAG graph.

    A plain TypedDict: nothing enforces the types at runtime. Each node
    returns a partial update, and LangGraph overwrites those keys, with
    no custom reducers. Nodes read optional keys with state.get() and
    treat a missing key as its zero value.

    Attributes:
        query: The text used for retrieval. CRAG rewrites replace it.
        original_query: The user's text as submitted.
        retrieved_docs: Chunks from the latest retrieval.
        retrieval_grade: The routing decision after grading.
        rewrite_count: How many query rewrites have run.
        final_response: The answer text, with the Sources footer.
        cache_hit: True if the answer came from the semantic cache.
        query_category: The router's category for the query.
        context_relevance: The grader's verdict on the latest
            retrieval, or None before grading.
        company_filter: Optional ticker that restricts dense search.
    """

    query: str
    original_query: str
    retrieved_docs: list[RetrievedDoc]
    retrieval_grade: Literal["relevant", "irrelevant"]
    rewrite_count: int
    final_response: str
    cache_hit: bool
    query_category: QueryCategory
    context_relevance: Literal["relevant", "irrelevant"] | None
    company_filter: str | None
