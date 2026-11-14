"""Pydantic request and response schemas for the SEC compliance RAG API."""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.agent.state import QueryCategory


class SourceCitation(BaseModel):
    cik: str = ""
    ticker: str
    company_name: str
    form_type: str
    filing_date: str
    period_of_report: str
    section_name: str
    accession_number: str
    chunk_text: str | None = None


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    session_id: str
    company_filter: str | None = None


class QueryResponse(BaseModel):
    response: str
    cache_hit: bool
    query_category: QueryCategory
    retrieved_doc_count: int
    retrieval_grade: str
    rewrite_count: int
    warnings: list[str]
    sources: list[SourceCitation] = Field(default_factory=list)
