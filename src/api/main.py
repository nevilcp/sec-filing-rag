"""
api/main.py

FastAPI application setup for SEC compliance RAG API.

Responsibilities:
  - Initialize the FastAPI application and configure CORS
  - Manage application lifespan (startup/shutdown logging, Qdrant init)
  - Attach access logging middleware for all HTTP requests

Dependencies:
  - fastapi
  - openinference-instrumentation-langchain
"""

from __future__ import annotations

import os
import time
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

from src.agent.graph import compile_graph
from src.api.routes import router
from src.rag.vector_store import get_filings_collection


class _AccessLogMiddleware(BaseHTTPMiddleware):
    """Logs every HTTP request to the ``sec_rag.api`` channel (JSON, file only)."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Record method, path, status code, and latency for every request.

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
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown lifecycle.

    On startup: initializes the Qdrant filings collection and compiles the
    LangGraph graph, attaching both to app.state so route handlers can
    access them without module-level globals.

    Yields:
        Control to the ASGI server while the application runs.
    """
    _app_log.info(
        "SEC RAG API starting up",
        extra={
            "extra": {
                "host": config.API_HOST,
                "port": config.API_PORT,
                "phoenix_enabled": os.getenv("PHOENIX_ENABLED", "false"),
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
    allow_origins=[
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:7860",
        "http://127.0.0.1",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:7860",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(_AccessLogMiddleware)

app.include_router(router)
