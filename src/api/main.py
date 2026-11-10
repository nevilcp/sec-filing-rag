"""FastAPI application setup for the SEC compliance RAG API.

Responsibilities:
  - Create the FastAPI app and configure CORS.
  - Manage the app lifespan: build the graph and metrics at startup.
  - Add access logging middleware for every HTTP request.
"""

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

# These imports follow configure_logging() on purpose: modules that
# log at import time must see the configured handlers. This is why
# E402 is ignored for this file.
from src.agent.graph import compile_graph
from src.api.routes import router
from src.rag.vector_store import get_filings_collection


class _AccessLogMiddleware(BaseHTTPMiddleware):
    """Log every HTTP request to the ``sec_rag.api`` channel."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Record method, path, status code and latency of a request.

        Args:
            request: The incoming HTTP request.
            call_next: The next middleware or route handler.

        Returns:
            The HTTP response from the downstream handler.
        """
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
    """Manage application startup and shutdown.

    On startup it ensures the Qdrant filings collection exists, then
    compiles the LangGraph graph and creates the metrics counters. The
    graph and counters are stored on app.state, so route handlers reach
    them without module-level globals.

    Args:
        app: The FastAPI application.

    Yields:
        Control to the ASGI server while the application runs.
    """
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
    get_filings_collection()
    app.state.graph = compile_graph()
    app.state.metrics = {
        "total_queries_served": 0,
        "cache_hits": 0,
        "rewrite_count_total": 0,
    }
    _app_log.info("SEC RAG API startup complete")
    yield
    _app_log.info("SEC RAG API shutting down")


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
