"""End-to-end evaluation runner for SEC compliance RAG API."""

from __future__ import annotations

import json
import re
import sys
import uuid
from pathlib import Path
from statistics import mean
from typing import Any

import requests

from src.core.llm_factory import get_classification_llm

try:
    from tests.eval.eval_dataset import EVAL_DATASET
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parent))
    from eval_dataset import EVAL_DATASET  # type: ignore

API_QUERY_URL = "http://localhost:8000/query"
REQUEST_TIMEOUT_SECONDS = 120
RESULTS_PATH = Path(__file__).resolve().parent / "results.json"
RELEVANCE_GATE = 0.75
BRACKETED_CITATION_PATTERN = re.compile(r"\[[^\]]+\]")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _llm_score(
    question: str,
    expected_topics: list[str],
    should_cite_regulation: bool,
    regulation_name: str | None,
    response_text: str,
) -> dict[str, float]:
    prompt = (
        "You are grading a compliance assistant response. "
        "Return JSON only with keys: relevance_score, citation_score, hallucination_score. "
        "Each score must be a float from 0 to 1.\n\n"
        "Scoring rubric:\n"
        "1) relevance_score: does response address expected topics?\n"
        "2) citation_score: if should_cite_regulation=true, does response include bracketed citation "
        "like [Rule 10b-5]? If false, score 1.0 unless citation behavior is harmful/misleading.\n"
        "3) hallucination_score: high score means no obvious fabrication/contradiction beyond question scope.\n\n"
        f"Question: {question}\n"
        f"Expected topics: {expected_topics}\n"
        f"should_cite_regulation: {should_cite_regulation}\n"
        f"regulation_name: {regulation_name}\n"
        f"Response: {response_text}\n"
    )
    raw_content = get_classification_llm().invoke(prompt).content
    if isinstance(raw_content, list):
        content = "".join(
            c.get("text", "") if isinstance(c, dict) else str(c)
            for c in raw_content
        )
    else:
        content = str(raw_content)
    
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\n|\n```$", "", content).strip()

    parsed: dict[str, Any]
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = {}

    relevance = min(max(_safe_float(parsed.get("relevance_score"), 0.0), 0.0), 1.0)
    citation = min(max(_safe_float(parsed.get("citation_score"), 0.0), 0.0), 1.0)
    hallucination = min(max(_safe_float(parsed.get("hallucination_score"), 0.0), 0.0), 1.0)

    if should_cite_regulation and citation == 0.0:
        citation = 1.0 if BRACKETED_CITATION_PATTERN.search(response_text) else 0.0
    return {
        "relevance_score": relevance,
        "citation_score": citation,
        "hallucination_score": hallucination,
    }


def _run_single_case(case: dict[str, Any], index: int) -> dict[str, Any]:
    payload = {
        "query": case["question"],
        "session_id": f"eval-{index}-{uuid.uuid4()}",
    }
    response = requests.post(API_QUERY_URL, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    result = response.json()

    response_text = str(result.get("response", ""))
    scores = _llm_score(
        question=str(case["question"]),
        expected_topics=list(case["expected_topics"]),
        should_cite_regulation=bool(case["should_cite_regulation"]),
        regulation_name=case.get("regulation_name"),
        response_text=response_text,
    )

    citation_source_score = 0.0
    if "sources" in result:
        sources = result["sources"]
        if isinstance(sources, list):
            if not sources:
                citation_source_score = 0.5
            elif any(s.get("ticker") for s in sources if isinstance(s, dict)):
                citation_source_score = 1.0

    return {
        "index": index,
        "question": case["question"],
        "expected_topics": case["expected_topics"],
        "should_cite_regulation": case["should_cite_regulation"],
        "regulation_name": case["regulation_name"],
        "response": response_text,
        "cache_hit": bool(result.get("cache_hit", False)),
        "query_category": str(result.get("query_category", "general")),
        "retrieval_grade": str(result.get("retrieval_grade", "irrelevant")),
        "retrieved_doc_count": int(result.get("retrieved_doc_count", 0)),
        "rewrite_count": int(result.get("rewrite_count", 0)),
        "warnings": list(result.get("warnings", [])),
        "citation_source_score": citation_source_score,
        **scores,
    }


def _print_summary(summary: dict[str, Any]) -> None:
    rows = [
        ("total_cases", summary["total_cases"]),
        ("mean_relevance_score", f"{summary['mean_relevance_score']:.3f}"),
        ("mean_citation_score", f"{summary['mean_citation_score']:.3f}"),
        ("mean_citation_source_score", f"{summary['mean_citation_source_score']:.3f}"),
        ("mean_hallucination_score", f"{summary['mean_hallucination_score']:.3f}"),
        ("cache_hit_rate", f"{summary['cache_hit_rate']:.3f}"),
        ("mean_rewrite_count", f"{summary['mean_rewrite_count']:.3f}"),
        ("mean_retrieved_doc_count", f"{summary['mean_retrieved_doc_count']:.3f}"),
    ]

    key_width = max(len(k) for k, _ in rows)
    val_width = max(len(str(v)) for _, v in rows)
    border = f"+-{'-' * key_width}-+-{'-' * val_width}-+"
    print(border)
    print(f"| {'metric'.ljust(key_width)} | {'value'.ljust(val_width)} |")
    print(border)
    for key, value in rows:
        print(f"| {str(key).ljust(key_width)} | {str(value).ljust(val_width)} |")
    print(border)


def main() -> int:
    results: list[dict[str, Any]] = []
    for idx, case in enumerate(EVAL_DATASET, start=1):
        results.append(_run_single_case(case, idx))

    relevance_scores = [item["relevance_score"] for item in results]
    citation_scores = [item["citation_score"] for item in results]
    citation_source_scores = [item["citation_source_score"] for item in results]
    hallucination_scores = [item["hallucination_score"] for item in results]
    cache_hits = [1.0 if item["cache_hit"] else 0.0 for item in results]
    rewrite_counts = [float(item["rewrite_count"]) for item in results]
    retrieved_counts = [float(item["retrieved_doc_count"]) for item in results]

    summary = {
        "total_cases": len(results),
        "mean_relevance_score": mean(relevance_scores) if relevance_scores else 0.0,
        "mean_citation_score": mean(citation_scores) if citation_scores else 0.0,
        "mean_citation_source_score": mean(citation_source_scores) if citation_source_scores else 0.0,
        "mean_hallucination_score": mean(hallucination_scores) if hallucination_scores else 0.0,
        "cache_hit_rate": mean(cache_hits) if cache_hits else 0.0,
        "mean_rewrite_count": mean(rewrite_counts) if rewrite_counts else 0.0,
        "mean_retrieved_doc_count": mean(retrieved_counts) if retrieved_counts else 0.0,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps({"summary": summary, "results": results}, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    _print_summary(summary)

    if summary["mean_relevance_score"] < RELEVANCE_GATE:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
