"""FastAPI application setup for SEC compliance RAG API."""

from __future__ import annotations

from contextlib import asynccontextmanager
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.agent.graph import compile_graph
from src.api.routes import router
from src.core import config
from src.rag.cache import SemanticCache
from src.rag.vector_store import get_filings_collection

if os.getenv("PHOENIX_ENABLED", "").lower() == "true":
    import phoenix as px


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_filings_collection()
    if os.getenv("PHOENIX_ENABLED", "").lower() == "true":
        # Phoenix server should be started externally via `phoenix serve`
        pass
    app.state.graph = compile_graph()
    app.state.cache = SemanticCache()
    app.state.metrics = {
        "total_queries_served": 0,
        "cache_hits": 0,
        "rewrite_count_total": 0,
    }
    yield


app = FastAPI(title="SEC Compliance RAG API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
