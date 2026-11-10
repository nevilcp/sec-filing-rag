"""FastAPI route handlers for the SEC compliance RAG API.

Responsibilities:
  - Expose HTTP endpoints for querying, ingestion, health and corpus
    status.
  - Wrap the LangGraph run in the input and output guardrails.
  - Keep the process-wide counters reported by /metrics.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from opentelemetry.trace import Status, StatusCode

from src.agent.state import RetrievedDoc, _doc_meta
from src.api.schemas import QueryRequest, QueryResponse, SourceCitation
from src.core import config
from src.core.logging import get_audit_logger, get_logger, get_rag_logger
from src.core.rate_limiter import get_embedding_rate_limiter
from src.core.tracing import chain_span, set_span_output
from src.guardrails.input_guard import validate_input
from src.guardrails.output_guard import validate_output
from src.rag.ingest import run_ingestion
from src.rag.vector_store import get_qdrant_client

logger = get_logger(__name__)
_rag_log = get_rag_logger()
_audit_log = get_audit_logger()
router = APIRouter()


def _run_ingestion_background() -> None:
    """Run the ingestion pipeline in a background task."""
    try:
        code = run_ingestion()
        logger.info(
            "Background ingestion completed",
            extra={"extra": {"exit_code": code}},
        )
    except Exception:
        logger.exception("Background ingestion failed")
        raise


def _build_initial_state(payload: QueryRequest) -> dict[str, Any]:
    """Return the starting graph state for a query request."""
    return {
        "query": payload.query,
        "original_query": payload.query,
        "retrieved_docs": [],
        "retrieval_grade": "irrelevant",
        "rewrite_count": 0,
        "final_response": "",
        "cache_hit": False,
        "query_category": "general",
        "company_filter": payload.company_filter,
    }


def _to_citations(docs: list[RetrievedDoc]) -> list[SourceCitation]:
    """Convert retrieved chunks into API source citations."""
    citations = []
    for doc in docs:
        meta = _doc_meta(doc)
        citations.append(
            SourceCitation(
                ticker=meta.get("ticker", ""),
                company_name=meta.get("company_name", ""),
                form_type=meta.get("form_type", ""),
                filing_date=meta.get("filing_date", ""),
                period_of_report=meta.get("period_of_report", ""),
                section_name=meta.get("section_name", ""),
                accession_number=meta.get("accession_number", ""),
                chunk_text=doc.get("text", ""),
            )
        )
    return citations


def _record_metrics(metrics: dict[str, int], cache_hit: bool, rewrite_count: int) -> None:
    """Add one served query to the process-wide counters."""
    metrics["total_queries_served"] += 1
    if cache_hit:
        metrics["cache_hits"] += 1
    metrics["rewrite_count_total"] += rewrite_count


@router.post("/query", response_model=QueryResponse)
def query_endpoint(payload: QueryRequest, request: Request) -> QueryResponse:
    """Answer a compliance query through the RAG graph.

    Validates the input, runs the LangGraph workflow, then sanitizes the
    answer with the output guard.

    Args:
        payload: The user query request.
        request: The FastAPI request object, which carries the compiled
            graph and the metrics on app.state.

    Returns:
        The answer with retrieved citations and any guardrail warnings.

    Raises:
        HTTPException: 400 if the input guard rejects the query.
    """
    with chain_span("query_endpoint", payload.query) as span:
        is_valid, reason = validate_input(payload.query)
        _audit_log.info(
            "input_guard_decision",
            extra={
                "extra": {
                    "valid": is_valid,
                    "reason": reason,
                    "session_id": payload.session_id,
                }
            },
        )
        if not is_valid:
            set_span_output(span, reason)
            span.set_status(Status(StatusCode.ERROR, reason))
            raise HTTPException(status_code=400, detail=reason)

        graph = request.app.state.graph
        result = graph.invoke(
            _build_initial_state(payload),
            config={"configurable": {"thread_id": payload.session_id}},
        )

        final_response = str(result.get("final_response", ""))
        sanitized_response, warnings = validate_output(final_response)
        _audit_log.info(
            "output_guard_decision",
            extra={"extra": {"warnings": warnings, "session_id": payload.session_id}},
        )

        cache_hit = bool(result.get("cache_hit", False))
        rewrite_count = int(result.get("rewrite_count", 0))
        query_category = str(result.get("query_category", "general"))
        retrieval_grade = str(result.get("retrieval_grade", "irrelevant"))
        docs = result.get("retrieved_docs", [])

        _record_metrics(request.app.state.metrics, cache_hit, rewrite_count)
        _rag_log.info(
            "query_complete",
            extra={
                "extra": {
                    "session_id": payload.session_id,
                    "query_category": query_category,
                    "cache_hit": cache_hit,
                    "retrieval_grade": retrieval_grade,
                    "retrieved_doc_count": len(docs),
                    "rewrite_count": rewrite_count,
                    "warnings": warnings,
                }
            },
        )

        response = QueryResponse(
            response=sanitized_response,
            cache_hit=cache_hit,
            query_category=query_category,
            retrieved_doc_count=len(docs),
            retrieval_grade=retrieval_grade,
            rewrite_count=rewrite_count,
            warnings=warnings,
            sources=_to_citations(docs),
        )
        set_span_output(span, sanitized_response)
        span.set_status(Status(StatusCode.OK))
        return response


@router.get("/health")
def health_endpoint() -> dict[str, str]:
    """Return a minimal liveness probe response."""
    return {"status": "ok", "version": "1.0.0"}


@router.get("/metrics")
def metrics_endpoint(request: Request) -> dict[str, Any]:
    """Return aggregate runtime metrics for the current server process.

    Reports the query count, cache hit rate, average CRAG rewrite count
    and the current usage of the rate limiter the endpoint tracks.

    Args:
        request: The FastAPI request object carrying app.state.metrics.

    Returns:
        A dict of metric names to their current values.
    """
    metrics = request.app.state.metrics
    total = int(metrics.get("total_queries_served", 0))
    cache_hits = int(metrics.get("cache_hits", 0))
    rewrites = int(metrics.get("rewrite_count_total", 0))

    cache_hit_rate = (cache_hits / total) if total else 0.0
    average_rewrite_count = (rewrites / total) if total else 0.0

    rl_stats = get_embedding_rate_limiter().stats()

    return {
        "total_queries_served": total,
        "cache_hit_rate": cache_hit_rate,
        "average_rewrite_count": average_rewrite_count,
        "embedding_rate_limits": {
            "rpm_used": rl_stats["rpm_used"],
            "rpm_limit": rl_stats["rpm_limit"],
            "tpm_used": rl_stats["tpm_used"],
            "tpm_limit": rl_stats["tpm_limit"],
            "rpd_used": rl_stats["rpd_used"],
            "rpd_limit": rl_stats["rpd_limit"],
        },
    }


@router.get("/corpus/status")
def corpus_status() -> dict[str, Any]:
    """Return the current state of the indexed document corpus.

    Reads the ingestion checkpoint for company and filing metadata, and
    queries Qdrant for the total chunk count.

    Returns:
        A dict containing total_chunks, indexed_companies, last_updated,
        and form_types_indexed.

    Raises:
        HTTPException: 503 if the corpus has not yet been initialized.
    """
    checkpoint_path = config.INGEST_CHECKPOINT_PATH
    if not os.path.exists(checkpoint_path):
        raise HTTPException(status_code=503, detail="Corpus not yet initialized")

    with open(checkpoint_path) as f:
        checkpoint = json.load(f)

    mtime = os.path.getmtime(checkpoint_path)
    last_updated = datetime.fromtimestamp(mtime, tz=UTC).isoformat()

    client = get_qdrant_client()
    try:
        count_result = client.count(collection_name=config.QDRANT_COLLECTION_NAME)
        total_chunks = count_result.count
    except Exception:
        total_chunks = 0

    return {
        "total_chunks": total_chunks,
        "indexed_companies": checkpoint.get("indexed_companies", []),
        "last_updated": last_updated,
        "form_types_indexed": ["10-K"],
    }


@router.post("/ingest/trigger")
def trigger_ingestion(background_tasks: BackgroundTasks) -> dict[str, str]:
    """Enqueue an asynchronous ingestion run and return immediately.

    The ingestion pipeline runs as a background task so the HTTP
    response is not blocked. Progress is logged to stdout.

    Args:
        background_tasks: FastAPI's background task runner.

    Returns:
        A dict confirming the ingestion job has been enqueued.
    """
    background_tasks.add_task(_run_ingestion_background)
    return {"status": "ingestion_started"}
