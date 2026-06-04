"""
agent/router.py

Routing helpers for conditional LangGraph edges.

Responsibilities:
  - Evaluate state transitions after cache checks and retrieval grading
  - Enforce CRAG max-rewrite safety limits to prevent infinite loops

Dependencies:
  - None
"""

from __future__ import annotations

from src.agent.state import ComplianceState
from src.core import config


def route_after_cache_check(state: ComplianceState) -> str:
    """Routes execution based on whether a valid semantic cache hit occurred.

    Args:
        state: Current graph state.

    Returns:
        The name of the next node ('respond' or 'process_input')
    """
    if state.get("cache_hit", False):
        return "respond"
    return "process_input"


def route_after_grade_retrieval(state: ComplianceState) -> str:
    """Evaluates if retrieval results are sufficient or require query rewriting.

    Args:
        state: Current graph state.

    Returns:
        'rewrite_query' if results are irrelevant and under the rewrite limit,
        otherwise 'respond'
    """
    is_irrelevant = state.get("retrieval_grade") == "irrelevant"
    rewrites = int(state.get("rewrite_count", 0))
    if is_irrelevant and rewrites < config.MAX_CRAG_REWRITES:
        return "rewrite_query"
    return "respond"
