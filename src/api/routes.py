"""
api/routes.py

FastAPI route handlers for SEC compliance RAG API.

Responsibilities:
  - Expose HTTP endpoints for RAG querying, ingestion, and system health
  - Inject input/output guardrails around the core LangGraph execution
  - Track endpoint-level metrics (cache hits, rewrites, latencies)

Dependencies:
  - fastapi
  - opentelemetry
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from src.api.schemas import QueryRequest, QueryResponse, SourceCitation
from src.core import config
from src.core.logging import get_audit_logger, get_logger, get_rag_logger
from src.core.rate_limiter import get_embedding_rate_limiter
from src.guardrails.input_guard import validate_input
from src.guardrails.output_guard import validate_output
from src.rag.ingest import run_ingestion
from src.rag.vector_store import get_qdrant_client

tracer = trace.get_tracer(__name__)

logger = get_logger(__name__)
_rag_log = get_rag_logger()
_audit_log = get_audit_logger()
router = APIRouter()


def _run_ingestion_background() -> None:
    """Executes the ingestion pipeline in a background thread"""
    try:
        code = run_ingestion()
        logger.info(
            "Background ingestion completed",
            extra={"extra": {"exit_code": code}},
        )
    except Exception:
        logger.exception("Background ingestion failed")
        raise


@router.post("/query", response_model=QueryResponse)
def query_endpoint(payload: QueryRequest, request: Request) -> QueryResponse:
    """Primary endpoint for executing a compliance RAG query.

    Wraps the LangGraph execution in input and output guardrails.

    Args:
        payload: The user query request.
        request: The FastAPI request object.

    Returns:
        The generated response with retrieved citations and safety warnings
    """
    with tracer.start_as_current_span(
        "query_endpoint",
        openinference_span_kind="chain",
    ) as span:
        span.set_input(payload.query)
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
            span.set_output(reason)
            span.set_status(Status(StatusCode.ERROR, reason))
            raise HTTPException(status_code=400, detail=reason)

        graph = request.app.state.graph
        result = graph.invoke(
            {
                "query": payload.query,
                "messages": [],
                "retrieved_docs": [],
                "retrieval_grade": "irrelevant",
                "rewrite_count": 0,
                "final_response": "",
                "cache_hit": False,
                "query_category": "general",
                "tool_calls_made": 0,
                "company_filter": payload.company_filter,
            },
            config={"configurable": {"thread_id": payload.session_id}},
        )

        final_response = str(result.get("final_response", ""))
        sanitized_response, warnings = validate_output(final_response)
        _audit_log.info(
            "output_guard_decision",
            extra={"extra": {"warnings": warnings, "session_id": payload.session_id}},
        )

        metrics = request.app.state.metrics
        metrics["total_queries_served"] += 1
        cache_hit = bool(result.get("cache_hit", False))
        if cache_hit:
            metrics["cache_hits"] += 1
        rewrite_count = int(result.get("rewrite_count", 0))
        metrics["rewrite_count_total"] += rewrite_count

        _rag_log.info(
            "query_complete",
            extra={
                "extra": {
                    "session_id": payload.session_id,
                    "query_category": result.get("query_category", "general"),
                    "cache_hit": cache_hit,
                    "retrieval_grade": result.get("retrieval_grade", "irrelevant"),
                    "retrieved_doc_count": len(result.get("retrieved_docs", [])),
                    "rewrite_count": rewrite_count,
                    "warnings": warnings,
                }
            },
        )

        docs = result.get("retrieved_docs", [])
        sources_list = []
        for doc in docs:
            meta = doc.get("metadata", {})
            sources_list.append(
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

        response = QueryResponse(
            response=sanitized_response,
            cache_hit=bool(result.get("cache_hit", False)),
            query_category=str(result.get("query_category", "general")),
            retrieved_doc_count=len(docs),
            retrieval_grade=str(result.get("retrieval_grade", "irrelevant")),
            rewrite_count=int(result.get("rewrite_count", 0)),
            warnings=warnings,
            sources=sources_list,
        )
        span.set_output(sanitized_response)
        span.set_status(Status(StatusCode.OK))
        return response


@router.get("/health")
def health_endpoint() -> dict[str, str]:
    """Return a minimal liveness probe response."""
    return {"status": "ok", "version": "1.0.0"}


@router.get("/metrics")
def metrics_endpoint(request: Request) -> dict[str, Any]:
    """Return aggregate runtime metrics for the current server process.

    Reports cumulative query counts, cache hit rate, average CRAG rewrite
    count, and current sliding-window usage for the embedding rate limiter.

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

    The ingestion pipeline runs in a BackgroundTasks thread so the HTTP
    response is not blocked. Progress is visible in logs/rag/pipeline.log.

    Args:
        background_tasks: FastAPI's background task runner.

    Returns:
        A dict confirming the ingestion job has been enqueued.
    """
    background_tasks.add_task(_run_ingestion_background)
    return {"status": "ingestion_started"}
