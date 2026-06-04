"""
api/schemas.py

Pydantic request/response schemas for the SEC compliance RAG API.

Responsibilities:
  - Define and validate the structure of incoming query requests.
  - Define the structure of outgoing query responses and citations.

Not responsible for:
  - Business logic or LLM invocation (see api/routes.py).
  - Guardrail enforcement (see guardrails/).

Dependencies:
  - pydantic >= 2.0
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SourceCitation(BaseModel):
    """A single document passage retrieved to support a generated answer.

    Fields correspond to EDGAR filing provenance metadata stored in Qdrant
    alongside each embedded chunk.
    """

    ticker: str
    company_name: str
    form_type: str
    filing_date: str
    period_of_report: str
    section_name: str
    accession_number: str
    chunk_text: str | None = None


class QueryRequest(BaseModel):
    """Validated inbound payload for the /query endpoint."""

    query: str = Field(..., min_length=1, max_length=2000)
    session_id: str
    company_filter: str | None = None


class QueryResponse(BaseModel):
    """Full response envelope returned by the /query endpoint.

    Includes the generated answer, pipeline diagnostic fields, guardrail
    warnings, and the list of retrieved source citations.
    """

    response: str
    cache_hit: bool
    query_category: str
    retrieved_doc_count: int
    retrieval_grade: str
    rewrite_count: int
    warnings: list[str]
    sources: list[SourceCitation] = Field(default_factory=list)
