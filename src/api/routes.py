"""FastAPI route handlers for SEC compliance RAG API."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request

from src.api.schemas import QueryRequest, QueryResponse, SourceCitation
from src.core import config
from src.rag.vector_store import get_qdrant_client
from src.core.logging import get_logger
from src.guardrails.input_guard import validate_input
from src.guardrails.output_guard import validate_output
from src.rag.ingest import run_ingestion

logger = get_logger(__name__)
router = APIRouter()


def _run_ingestion_background() -> None:
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
    is_valid, reason = validate_input(payload.query)
    if not is_valid:
        raise HTTPException(status_code=422, detail=reason)

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

    metrics = request.app.state.metrics
    metrics["total_queries_served"] += 1
    if bool(result.get("cache_hit", False)):
        metrics["cache_hits"] += 1
    metrics["rewrite_count_total"] += int(result.get("rewrite_count", 0))

    docs = result.get("retrieved_docs", [])
    sources_list = []
    for doc in docs:
        meta = doc.get("metadata", {})
        sources_list.append(SourceCitation(
            ticker=meta.get("ticker", ""),
            company_name=meta.get("company_name", ""),
            form_type=meta.get("form_type", ""),
            filing_date=meta.get("filing_date", ""),
            period_of_report=meta.get("period_of_report", ""),
            section_name=meta.get("section_name", ""),
            accession_number=meta.get("accession_number", "")
        ))

    return QueryResponse(
        response=sanitized_response,
        cache_hit=bool(result.get("cache_hit", False)),
        query_category=str(result.get("query_category", "general")),
        retrieved_doc_count=len(docs),
        retrieval_grade=str(result.get("retrieval_grade", "irrelevant")),
        rewrite_count=int(result.get("rewrite_count", 0)),
        warnings=warnings,
        sources=sources_list,
    )


@router.get("/health")
def health_endpoint() -> dict[str, str]:
    return {"status": "ok", "version": "1.0.0"}


@router.get("/metrics")
def metrics_endpoint(request: Request) -> dict[str, Any]:
    metrics = request.app.state.metrics
    total = int(metrics.get("total_queries_served", 0))
    cache_hits = int(metrics.get("cache_hits", 0))
    rewrites = int(metrics.get("rewrite_count_total", 0))

    cache_hit_rate = (cache_hits / total) if total else 0.0
    average_rewrite_count = (rewrites / total) if total else 0.0

    from src.core.rate_limiter import get_embedding_rate_limiter
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
    checkpoint_path = config.INGEST_CHECKPOINT_PATH
    if not os.path.exists(checkpoint_path):
        raise HTTPException(status_code=503, detail="Corpus not yet initialized")
    
    with open(checkpoint_path, "r") as f:
        checkpoint = json.load(f)
    
    mtime = os.path.getmtime(checkpoint_path)
    last_updated = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
    
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
        "form_types_indexed": ["10-K"]
    }


@router.post("/ingest/trigger")
def trigger_ingestion(background_tasks: BackgroundTasks) -> dict[str, str]:
    background_tasks.add_task(_run_ingestion_background)
    return {"status": "ingestion_started"}
