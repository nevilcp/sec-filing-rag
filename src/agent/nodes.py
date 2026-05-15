"""LangGraph node implementations for SEC compliance workflow."""

from __future__ import annotations

import json
from typing import Any

from src.agent.state import ComplianceState
from src.core import config
from src.core.llm_factory import get_primary_llm, get_premium_llm, get_classification_llm
from src.core.logging import get_logger
from src.rag.cache import SemanticCache
from src.rag.retriever import retrieve

logger = get_logger(__name__)
_CACHE: SemanticCache | None = None

_CATEGORY_INSTRUCTIONS = {
    "compliance_check": (
        "Focus on regulatory obligations, prohibited conduct, reporting thresholds, and "
        "compliance controls."
    ),
    "document_analysis": (
        "Focus on interpreting what the supplied regulatory excerpts say and how they "
        "apply to the described document facts."
    ),
    "cross_regulation": (
        "Compare and reconcile requirements across multiple cited regulations, noting "
        "areas of overlap and differences."
    ),
    "general": (
        "Provide a clear regulation-grounded explanation relevant to the user question."
    ),
    "cross_company": (
        "Compare disclosures across the specified companies. Note similarities, differences, "
        "and any company-specific risk language. Cite each company by ticker and filing date."
    ),
    "temporal": (
        "Analyze how the specified company's disclosure on this topic has changed across "
        "filing periods. Note additions, removals, and material rewording. Cite by period."
    ),
}


def _cache() -> SemanticCache:
    global _CACHE
    if _CACHE is None:
        _CACHE = SemanticCache()
    return _CACHE


def process_input(state: ComplianceState) -> dict[str, Any]:
    query_text = state["query"]
    prompt = (
        "Classify the user query into exactly one category from this set: "
        '["compliance_check", "document_analysis", "cross_regulation", "general", "cross_company", "temporal"]. '
        "Return only JSON like {\"query_category\": \"...\"}.\n\n"
        f"Query: {query_text}"
    )
    response = get_primary_llm().invoke(prompt)
    content = str(response.content).strip()

    category = "general"
    try:
        parsed = json.loads(content)
        candidate = str(parsed.get("query_category", "general")).strip()
        if candidate in _CATEGORY_INSTRUCTIONS:
            category = candidate
    except json.JSONDecodeError:
        lowered = content.lower()
        for option in _CATEGORY_INSTRUCTIONS:
            if option in lowered:
                category = option
                break

    logger.info(
        "process_input complete",
        extra={"extra": {"query_category": category}},
    )
    return {"query_category": category}


def cache_check(state: ComplianceState) -> dict[str, Any]:
    cached = _cache().lookup(state["query"])
    if cached is not None:
        logger.info("cache_check hit")
        return {"cache_hit": True, "final_response": cached}
    logger.info("cache_check miss")
    return {"cache_hit": False}


def retrieve_and_reason(state: ComplianceState) -> dict[str, Any]:
    query_text = state["query"]
    docs = retrieve(query_text)

    formatted_docs = []
    for doc in docs:
        ticker   = doc.get('metadata', {}).get('ticker', '')
        form     = doc.get('metadata', {}).get('form_type', '')
        date     = doc.get('metadata', {}).get('filing_date', '')
        section  = doc.get('metadata', {}).get('section_name', '')
        formatted_docs.append(f"[{ticker} {form} {date} — {section}] {doc.get('text', '')}")
    context = "\n---\n".join(formatted_docs)
    
    category = state.get("query_category", "general")
    category_instruction = _CATEGORY_INSTRUCTIONS.get(category, _CATEGORY_INSTRUCTIONS["general"])

    prompt = (
        "System instructions:\n"
        "1) You are an SEC/CFR compliance assistant and do not provide legal advice.\n"
        "2) Base your answer only on the provided context.\n"
        "3) Cite sources inline using [TICKER FORM_TYPE FILING_DATE — SECTION] labels, "
        "e.g. [AAPL 10-K 2024-10-31 — risk_factors].\n"
        f"4) Task focus: {category_instruction}\n\n"
        f"User query:\n{query_text}\n\n"
        f"Context:\n{context if context else 'No context retrieved.'}\n"
    )
    response = get_premium_llm().invoke(prompt)
    final_response = str(response.content)

    logger.info(
        "retrieve_and_reason complete",
        extra={"extra": {"retrieved_count": len(docs)}},
    )
    return {
        "retrieved_docs": docs,
        "final_response": final_response,
        "tool_calls_made": int(state.get("tool_calls_made", 0)) + 1,
    }


def grade_retrieval(state: ComplianceState) -> dict[str, Any]:
    query_text = state["query"]
    docs = state.get("retrieved_docs", [])
    preview = "\n".join(doc.get("text", "")[:300] for doc in docs[:5])
    prompt = (
        'Return exactly one word: "relevant" or "irrelevant". '
        "Mark relevant only if retrieved context directly helps answer the query.\n\n"
        f"Query: {query_text}\n\nRetrieved preview:\n{preview}"
    )
    result = str(get_primary_llm().invoke(prompt).content).strip().lower()
    grade = "relevant" if "relevant" in result and "irrelevant" not in result else "irrelevant"
    logger.info("grade_retrieval complete", extra={"extra": {"retrieval_grade": grade}})
    return {"retrieval_grade": grade}


def rewrite_query(state: ComplianceState) -> dict[str, Any]:
    rewrite_count = int(state.get("rewrite_count", 0))
    if rewrite_count >= config.MAX_CRAG_REWRITES:
        logger.info("rewrite_query max reached; forcing relevant")
        return {"retrieval_grade": "relevant"}

    prompt = (
        "Rewrite the query using SEC/CFR-specific terminology to improve legal retrieval. "
        "Keep intent unchanged. Return only rewritten query text.\n\n"
        f"Original query: {state['query']}"
    )
    rewritten_query = str(get_primary_llm().invoke(prompt).content).strip()
    logger.info("rewrite_query complete", extra={"extra": {"rewrite_count": rewrite_count + 1}})
    return {
        "query": rewritten_query or state["query"],
        "rewrite_count": rewrite_count + 1,
    }


def respond(state: ComplianceState) -> dict[str, Any]:
    response_text = state.get("final_response", "")
    if not state.get("cache_hit", False):
        _cache().store(state["query"], response_text)
    docs = state.get("retrieved_docs", [])
    sources_text = ""
    if docs:
        sources_list = []
        for doc in docs:
            meta = doc.get("metadata", {})
            ticker = meta.get("ticker", "")
            form_type = meta.get("form_type", "")
            filing_date = meta.get("filing_date", "")
            section_name = meta.get("section_name", "")
            accession_number = meta.get("accession_number", "")
            sources_list.append(f"- {ticker} | {form_type} | {filing_date} | {section_name} (Accession: {accession_number})")
        sources_text = "\n\nSources:\n" + "\n".join(sources_list)

    disclaimer = (
        "\n\nDisclaimer: This response is for informational purposes only and does not "
        "constitute legal advice."
    )
    final = f"{response_text}{sources_text}{disclaimer}"
    logger.info("respond complete")
    return {"final_response": final}
