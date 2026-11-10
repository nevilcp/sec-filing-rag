"""Routing helpers for conditional LangGraph edges.

Responsibilities:
  - Choose the next node after the cache check and after retrieval
    grading.
  - Cap CRAG query rewrites at config.MAX_CRAG_REWRITES so the loop
    always ends.
"""

from __future__ import annotations

from src.agent.state import ComplianceState
from src.core import config


def route_after_cache_check(state: ComplianceState) -> str:
    """Choose the next node after the semantic cache check.

    Args:
        state: Current graph state.

    Returns:
        'respond' on a cache hit, otherwise 'process_input'.
    """
    if state.get("cache_hit", False):
        return "respond"
    return "process_input"


def route_after_grade_retrieval(state: ComplianceState) -> str:
    """Choose between rewriting the query and answering.

    Args:
        state: Current graph state.

    Returns:
        'rewrite_query' if the results are irrelevant and the rewrite
        limit is not reached, otherwise 'respond'.
    """
    is_irrelevant = state.get("retrieval_grade") == "irrelevant"
    rewrites = int(state.get("rewrite_count", 0))
    if is_irrelevant and rewrites < config.MAX_CRAG_REWRITES:
        return "rewrite_query"
    return "respond"
