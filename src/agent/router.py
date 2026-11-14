"""Routing helpers for conditional LangGraph edges."""

from __future__ import annotations

from src.agent.state import ComplianceState
from src.core import config


def route_after_cache_check(state: ComplianceState) -> str:
    if state.get("cache_hit", False):
        return "respond"
    return "process_input"


def route_after_grade_retrieval(state: ComplianceState) -> str:
    is_irrelevant = state.get("retrieval_grade") == "irrelevant"
    rewrites = int(state.get("rewrite_count", 0))
    if is_irrelevant and rewrites < config.MAX_CRAG_REWRITES:
        return "rewrite_query"
    return "respond"
