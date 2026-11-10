"""LangGraph node implementations for the SEC compliance workflow.

Responsibilities:
  - Implement the business logic of each LangGraph node.
  - Classify queries so the answer style fits the question.
  - Run hybrid retrieval, relevance grading and query rewriting.
  - Write grounded final answers from the retrieved context.
"""

from __future__ import annotations

import json
import string
from typing import Any

from src.agent.prompts import CATEGORY_INSTRUCTIONS, grader_prompt, respond_prompt, rewrite_prompt, router_prompt
from src.agent.state import ComplianceState, RetrievedDoc, _doc_meta
from src.core import config
from src.core.lazy import locked_singleton
from src.core.llm_factory import (
    get_generation_llm,
    get_grader_llm,
    get_rewriter_llm,
    get_router_llm,
    invoke_with_rate_limit,
)
from src.core.logging import get_logger
from src.core.rate_limiter import get_generation_limiter, get_router_limiter, get_workhorse_limiter
from src.rag.cache import SemanticCache
from src.rag.retriever import retrieve

logger = get_logger(__name__)


@locked_singleton
def _cache() -> SemanticCache:
    """Return the process-lifetime SemanticCache singleton.

    Deferred initialization avoids opening the Qdrant connection at
    import time, which would fail if the vector store is unavailable.

    Returns:
        The shared SemanticCache instance.
    """
    return SemanticCache()


def _original_query(state: ComplianceState) -> str:
    """Return the user's query as submitted, before any CRAG rewrite."""
    return state.get("original_query") or state["query"]


def process_input(state: ComplianceState) -> dict[str, Any]:
    """Classify the query into a category that selects the answer style.

    Falls back to 'general' when the model reply cannot be parsed.

    Args:
        state: Current graph state containing the user query.

    Returns:
        A state update holding the query_category.

    Raises:
        RuntimeError: If GROQ_API_KEY is not set.
        RateLimitExhausted: If the prompt can never fit the router
            model's token limit.
    """
    query_text = state["query"]
    prompt = router_prompt(query_text)
    content = invoke_with_rate_limit(
        get_router_llm(),
        prompt,
        get_router_limiter(),
    ).strip()

    category = "general"
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, dict):
        candidate = str(parsed.get("query_category", "general")).strip()
        if candidate in CATEGORY_INSTRUCTIONS:
            category = candidate
    else:
        lowered = content.lower()
        for option in CATEGORY_INSTRUCTIONS:
            if option in lowered:
                category = option
                break

    logger.info(
        "process_input complete",
        extra={"extra": {"query_category": category}},
    )
    return {"query_category": category}


def cache_check(state: ComplianceState) -> dict[str, Any]:
    """Look up the original query in the semantic cache.

    Args:
        state: Current graph state.

    Returns:
        A state update with cache_hit, plus final_response on a hit.
    """
    cached = _cache().lookup(_original_query(state))
    if cached is not None:
        logger.info("cache_check hit")
        return {"cache_hit": True, "final_response": cached}
    logger.info("cache_check miss")
    return {"cache_hit": False}


def retrieve_and_reason(state: ComplianceState) -> dict[str, Any]:
    """Retrieve documents and grade how relevant they are.

    The context counts as relevant only if it passes both a score check
    (mean and top-1 reranker score against config thresholds) and an LLM
    judge that reads previews of the first five documents.

    Args:
        state: Current graph state.

    Returns:
        A state update with retrieved_docs and context_relevance.

    Raises:
        RuntimeError: If GROQ_API_KEY is not set.
        FileNotFoundError: If the BM25 index has not been built.
    """
    query_text = state["query"]
    docs = retrieve(query_text, company_filter=state.get("company_filter"))

    scores = [float(doc.get("score", 0.0)) for doc in docs]
    if scores:
        mean_score = sum(scores) / len(scores)
        top1_score = max(scores)
    else:
        mean_score = 0.0
        top1_score = 0.0

    passes_math = (mean_score >= config.RELEVANCE_MEAN_THRESHOLD) and (top1_score >= config.RELEVANCE_TOP1_THRESHOLD)

    passes_semantic = False
    if docs:
        formatted_preview = []
        for doc in docs[:5]:
            meta = _doc_meta(doc)
            ticker = meta.get("ticker", "UNKNOWN")
            date = meta.get("filing_date", "UNKNOWN")
            text = doc.get("text", "")[:400]
            formatted_preview.append(f"[{ticker} | {date}] {text}...")
        preview = "\n\n".join(formatted_preview)

        prompt = grader_prompt(query_text, preview)
        result = invoke_with_rate_limit(
            get_grader_llm(),
            prompt,
            get_workhorse_limiter(),
        )
        words = result.strip().lower().split()
        verdict = words[0].strip(string.punctuation) if words else ""
        passes_semantic = verdict == "relevant"

    context_relevance = "relevant" if (passes_math and passes_semantic) else "irrelevant"

    logger.info(
        "retrieve_and_reason complete",
        extra={
            "extra": {
                "retrieved_count": len(docs),
                "mean_score": round(mean_score, 4),
                "top1_score": round(top1_score, 4),
                "passes_math": passes_math,
                "passes_semantic": passes_semantic,
                "context_relevance": context_relevance,
            }
        },
    )
    return {
        "retrieved_docs": docs,
        "context_relevance": context_relevance,
    }


def grade_retrieval(state: ComplianceState) -> dict[str, Any]:
    """Copy context_relevance into retrieval_grade for the router.

    Args:
        state: Current graph state.

    Returns:
        A state update holding the retrieval_grade.
    """
    context_relevance = state.get("context_relevance", "irrelevant")
    logger.info(
        "grade_retrieval complete",
        extra={"extra": {"retrieval_grade": context_relevance, "source": "score_based"}},
    )
    return {"retrieval_grade": context_relevance}


def rewrite_query(state: ComplianceState) -> dict[str, Any]:
    """Rewrite the query to improve retrieval.

    Runs after a failed retrieval. The rewrite uses formal SEC
    terminology and explicit entity names. Once the limit
    config.MAX_CRAG_REWRITES is reached, it only sets retrieval_grade
    to 'relevant' so the graph moves on to respond.

    Args:
        state: Current graph state.

    Returns:
        A state update with the rewritten query and the incremented
        rewrite_count.

    Raises:
        RuntimeError: If GROQ_API_KEY is not set.
    """
    rewrite_count = int(state.get("rewrite_count", 0))
    if rewrite_count >= config.MAX_CRAG_REWRITES:
        logger.info("rewrite_query max reached; forcing relevant")
        return {"retrieval_grade": "relevant"}

    prompt = rewrite_prompt(state["query"])
    rewritten_query = invoke_with_rate_limit(
        get_rewriter_llm(),
        prompt,
        get_workhorse_limiter(),
    ).strip()
    logger.info(
        "rewrite_query complete",
        extra={
            "extra": {
                "rewrite_count": rewrite_count + 1,
                "rewritten_query": rewritten_query,
            }
        },
    )
    return {
        "query": rewritten_query or state["query"],
        "rewrite_count": rewrite_count + 1,
        "retrieved_docs": [],
        "final_response": "",
        "context_relevance": None,
    }


def _format_context(docs: list[RetrievedDoc]) -> str:
    """Join retrieved chunks into the context block of the prompt."""
    formatted_docs = []
    for doc in docs:
        meta = _doc_meta(doc)
        ticker = meta.get("ticker", "")
        form = meta.get("form_type", "")
        date = meta.get("filing_date", "")
        section = meta.get("section_name", "")
        formatted_docs.append(f"[{ticker} {form} {date} — {section}] {doc.get('text', '')}")
    return "\n---\n".join(formatted_docs)


def _format_sources(docs: list[RetrievedDoc]) -> str:
    """Return the Sources footer for an answer, or an empty string."""
    if not docs:
        return ""
    sources_list = []
    for doc in docs:
        meta = _doc_meta(doc)
        ticker = meta.get("ticker", "")
        form_type = meta.get("form_type", "")
        filing_date = meta.get("filing_date", "")
        section_name = meta.get("section_name", "")
        accession_number = meta.get("accession_number", "")
        sources_list.append(
            f"- {ticker} | {form_type} | {filing_date} | {section_name} (Accession: {accession_number})"
        )
    return "\n\nSources:\n" + "\n".join(sources_list)


def respond(state: ComplianceState) -> dict[str, Any]:
    """Write the final answer for the query.

    A cache hit returns the cached text unchanged. An irrelevant grade
    returns a fixed "could not find" message. Otherwise the generation
    model answers from the retrieved context in the style of the query
    category, a Sources footer is appended and the result is cached
    under the original query.

    Args:
        state: Current graph state.

    Returns:
        A state update with final_response, which includes the Sources
        footer, and an empty retrieved_docs for the irrelevant case.

    Raises:
        RuntimeError: If GEMINI_API_KEY is not set.
    """
    query_text = _original_query(state)
    docs = state.get("retrieved_docs", [])

    if state.get("cache_hit", False):
        # The cache holds the final text, Sources included.
        logger.info("respond using cached final_response")
        return {"final_response": state.get("final_response", "")}

    if state.get("retrieval_grade") == "irrelevant":
        logger.info("respond short-circuit - retrieval grade is irrelevant after rewrites")
        return {
            "final_response": "I could not find sufficient information in the provided SEC filings to answer your question. Please try rephrasing or broadening your search.",
            "retrieved_docs": [],
        }

    context = _format_context(docs)
    category = state.get("query_category", "general")

    prompt = respond_prompt(category, context, query_text)
    response_text = invoke_with_rate_limit(
        get_generation_llm(),
        prompt,
        get_generation_limiter(),
    )

    sources_text = _format_sources(docs)

    final = f"{response_text}{sources_text}"
    _cache().store(query_text, final)
    logger.info("respond complete")
    return {"final_response": final}
