"""FastAPI application setup for the SEC compliance RAG API."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from src.core import config
from src.core.logging import (
    configure_logging,
    get_api_logger,
    get_app_logger,
)

configure_logging()
_app_log = get_app_logger()
_api_log = get_api_logger()

# Configure handlers before importing modules that may log during import.
from src.agent.graph import compile_graph
from src.agent.nodes import _cache
from src.api.routes import router
from src.rag.vector_store import get_filings_collection, get_qdrant_client


class _AccessLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        _api_log.info(
            "http_request",
            extra={
                "extra": {
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "latency_ms": latency_ms,
                    "client": request.client.host if request.client else None,
                }
            },
        )
        return response


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize the filings collection, compiled graph, and process metrics at startup."""
    _app_log.info(
        "SEC RAG API starting up",
        extra={
            "extra": {
                "host": config.API_HOST,
                "port": config.API_PORT,
                "phoenix_enabled": config.PHOENIX_ENABLED,
            }
        },
    )
    client = get_qdrant_client()
    try:
        get_filings_collection()
        app.state.graph = compile_graph()
        app.state.metrics = {
            "total_queries_served": 0,
            "cache_hits": 0,
            "rewrite_count_total": 0,
        }
        _app_log.info("SEC RAG API startup complete")
        yield
    finally:
        _app_log.info("SEC RAG API shutting down")
        client.close()
        _cache.cache_clear()
        get_filings_collection.cache_clear()
        get_qdrant_client.cache_clear()


app = FastAPI(title="SEC Compliance RAG API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(_AccessLogMiddleware)

app.include_router(router)
