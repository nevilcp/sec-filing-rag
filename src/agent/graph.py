"""
agent/graph.py

LangGraph assembly for the SEC compliance CRAG workflow.

Responsibilities:
  - Define the StateGraph topology for the CRAG agent pipeline.
  - Wire conditional routing logic to execution nodes.
  - Compile the graph with a MemorySaver checkpointer for multi-turn
    session continuity.

Not responsible for:
  - Node business logic (see agent/nodes.py).
  - Routing predicate definitions (see agent/router.py).

Dependencies:
  - langgraph
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agent.nodes import (
    cache_check,
    grade_retrieval,
    process_input,
    respond,
    retrieve_and_reason,
    rewrite_query,
)
from src.agent.router import route_after_cache_check, route_after_grade_retrieval
from src.agent.state import ComplianceState


def compile_graph() -> CompiledStateGraph:
    """Compile the executable LangGraph state machine for the CRAG workflow.

    Returns:
        A CompiledGraph instance backed by MemorySaver, ready for
        invocation via graph.invoke().
    """
    graph_builder = StateGraph(ComplianceState)

    graph_builder.add_node("process_input", process_input)
    graph_builder.add_node("cache_check", cache_check)
    graph_builder.add_node("retrieve_and_reason", retrieve_and_reason)
    graph_builder.add_node("grade_retrieval", grade_retrieval)
    graph_builder.add_node("rewrite_query", rewrite_query)
    graph_builder.add_node("respond", respond)

    graph_builder.add_edge(START, "cache_check")

    graph_builder.add_conditional_edges(
        "cache_check",
        route_after_cache_check,
        {
            "respond": "respond",
            "process_input": "process_input",
        },
    )

    graph_builder.add_edge("process_input", "retrieve_and_reason")

    graph_builder.add_edge("retrieve_and_reason", "grade_retrieval")
    graph_builder.add_conditional_edges(
        "grade_retrieval",
        route_after_grade_retrieval,
        {
            "rewrite_query": "rewrite_query",
            "respond": "respond",
        },
    )
    graph_builder.add_edge("rewrite_query", "retrieve_and_reason")
    graph_builder.add_edge("respond", END)

    return graph_builder.compile(checkpointer=MemorySaver())
