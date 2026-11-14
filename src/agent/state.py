"""State contract for the compliance LangGraph workflow."""

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
    text: str
    score: float
    metadata: dict[str, Any]


def _doc_meta(doc: RetrievedDoc) -> dict[str, Any]:
    return doc.get("metadata", {})


class ComplianceState(TypedDict):
    """Graph state with partial updates and no custom reducers or runtime type validation."""

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
