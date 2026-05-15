"""State contract for the compliance LangGraph workflow."""

from __future__ import annotations

from typing import Literal, TypedDict


class ComplianceState(TypedDict):
    query: str
    messages: list
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
    tool_calls_made: int
