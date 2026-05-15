"""Request/response schemas for API endpoints."""

from __future__ import annotations

from pydantic import BaseModel


class SourceCitation(BaseModel):
    ticker: str
    company_name: str
    form_type: str
    filing_date: str
    period_of_report: str
    section_name: str
    accession_number: str


class QueryRequest(BaseModel):
    query: str
    session_id: str
    company_filter: str | None = None


class QueryResponse(BaseModel):
    response: str
    cache_hit: bool
    query_category: str
    retrieved_doc_count: int
    retrieval_grade: str
    rewrite_count: int
    warnings: list[str]
    sources: list[SourceCitation] = []
