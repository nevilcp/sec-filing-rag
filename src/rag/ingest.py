"""
rag/ingest.py

EDGAR ingestion pipeline with checkpointing, chunking, embedding, and BM25 rebuilding.

Responsibilities:
  - Orchestrate the end-to-end data ingestion workflow for SEC filings
  - Manage idempotent execution via disk-backed checkpoints
  - Coordinate embedding generation and vector store updates
  - Trigger sparse index rebuilds upon completion

Dependencies:
  - langchain_text_splitters
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ModuleNotFoundError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.core import config
from src.core.corpus import get_corpus_companies, resolve_cik
from src.core.embeddings import embed_documents
from src.core.logging import get_logger
from src.rag import bm25_index, edgar_client, section_extractor
from src.rag.vector_store import (
    get_filings_collection,
    get_qdrant_client,
    upsert_chunks,
)

logger = get_logger(__name__)


def _today_date_str() -> str:
    """Return today's UTC date as an ISO 8601 string (e.g. '2024-12-31')."""
    return datetime.now(UTC).date().isoformat()


def _load_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    """Load or initialise the ingestion checkpoint from disk.

    Resets the date field to today if the on-disk checkpoint was written
    on a previous day, ensuring daily re-ingestion runs cleanly.

    Args:
        checkpoint_path: Path to the JSON checkpoint file.

    Returns:
        A dict with 'date', 'processed_files', and 'indexed_companies' keys.
    """
    default = {
        "date": _today_date_str(),
        "processed_files": [],
        "indexed_companies": [],
    }
    if not checkpoint_path.exists():
        return default

    content = checkpoint_path.read_text(encoding="utf-8").strip()
    if not content:
        return default

    loaded = json.loads(content)
    checkpoint = {
        "date": str(loaded.get("date", default["date"])),
        "processed_files": list(loaded.get("processed_files", [])),
        "indexed_companies": list(loaded.get("indexed_companies", [])),
    }

    if checkpoint["date"] != _today_date_str():
        checkpoint["date"] = _today_date_str()
    return checkpoint


def _save_checkpoint(checkpoint_path: Path, checkpoint: dict[str, Any]) -> None:
    """Atomically write the checkpoint dict to disk as UTF-8 JSON.

    Args:
        checkpoint_path: Destination path for the JSON file.
        checkpoint: The checkpoint state dict to persist.
    """
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text(json.dumps(checkpoint, indent=2, ensure_ascii=True), encoding="utf-8")


def _iter_sub_batches(items: list[Any], size: int) -> list[list[Any]]:
    """Split a list into contiguous sub-batches of at most `size` items.

    Args:
        items: The input list to partition.
        size: Maximum number of items per batch.

    Returns:
        A list of sub-lists, each no longer than `size`.
    """
    return [items[i : i + size] for i in range(0, len(items), size)]


def run_ingestion(tickers: list[str] | None = None) -> int:
    """Execute the end-to-end EDGAR ingestion pipeline.

    For each company in the corpus, fetches recent 10-K filings from
    EDGAR, extracts configured sections, splits them into chunks, embeds
    them in batches, and upserts into Qdrant. Idempotent: filings already
    recorded in the checkpoint file are skipped. After all upserts, the
    BM25 sparse index is rebuilt from the full Qdrant corpus.

    Args:
        tickers: Optional list of ticker symbols to restrict ingestion to.
            If None, all companies in KNOWN_TICKERS are processed.

    Returns:
        Exit code 0 on success.
    """
    checkpoint_path = Path(config.INGEST_CHECKPOINT_PATH)
    checkpoint = _load_checkpoint(checkpoint_path)
    _save_checkpoint(checkpoint_path, checkpoint)

    processed_files = set(checkpoint["processed_files"])

    companies = get_corpus_companies()
    if tickers:
        lower_tickers = {t.lower() for t in tickers}
        companies = [c for c in companies if c.get("ticker", "").lower() in lower_tickers]

    resolved_companies = []
    for comp in companies:
        cik = resolve_cik(comp["ticker"])
        if cik:
            resolved_companies.append({**comp, "cik": cik})

    filing_queue = []
    for comp in resolved_companies:
        filings = edgar_client.get_recent_filings(
            comp["cik"], form_type="10-K", max_results=len(config.EDGAR_FISCAL_YEARS)
        )
        for filing in filings:
            if filing["accession_number"] not in processed_files:
                filing_queue.append((comp["ticker"], comp["cik"], comp.get("company_name", ""), filing))

    if not filing_queue:
        logger.info("No new filings to process.")
        return 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )

    for ticker, cik, company_name, filing in filing_queue:
        accession_number = filing["accession_number"]

        html_path = edgar_client.stage_filing(
            cik=cik,
            ticker=ticker,
            accession_number=accession_number,
            filename=filing["primary_document"],
            staging_dir=Path(config.EDGAR_STAGING_PATH),
        )
        if not html_path or not Path(html_path).exists():
            continue

        html_content = Path(html_path).read_text(encoding="utf-8")

        sections = section_extractor.extract_sections(html_content)
        if not sections:
            continue

        chunks = []
        chunk_index = 0
        for section_name, section_text in sections.items():
            text_chunks = splitter.split_text(section_text)
            for text_chunk in text_chunks:
                clean_text = text_chunk.strip()
                if not clean_text:
                    continue

                chunks.append(
                    {
                        "id": f"{accession_number}::{section_name}::{chunk_index}",
                        "text": clean_text,
                        "metadata": {
                            "ticker": ticker,
                            "company_name": company_name,
                            "cik": cik,
                            "form_type": filing.get("form_type", ""),
                            "filing_date": filing.get("filing_date", ""),
                            "period_of_report": filing.get("period_of_report", ""),
                            "section_name": section_name,
                            "accession_number": accession_number,
                            "chunk_index": chunk_index,
                        },
                    }
                )
                chunk_index += 1

        if not chunks:
            continue

        batches = _iter_sub_batches(chunks, config.EMBEDDING_BATCH_SIZE)
        total_batches = len(batches)

        for batch_num, batch in enumerate(batches, 1):
            logger.debug("Embedding batch %d/%d", batch_num, total_batches)
            texts = [chunk["text"] for chunk in batch]
            try:
                embeddings = embed_documents(texts)
            except Exception:
                logger.exception(
                    "Embedding failed for batch %d/%d (%s). Skipping batch.",
                    batch_num,
                    total_batches,
                    ticker,
                )
                continue

            if len(embeddings) != len(batch):
                logger.warning(
                    "Batch %d/%d: expected %d embeddings, got %d. Skipping.",
                    batch_num,
                    total_batches,
                    len(batch),
                    len(embeddings),
                )
                continue

            for i, chunk in enumerate(batch):
                chunk["embedding"] = embeddings[i]

        valid_chunks = [c for c in chunks if c.get("embedding") is not None]
        if valid_chunks:
            upsert_chunks(valid_chunks)

        checkpoint["processed_files"].append(accession_number)
        if ticker not in checkpoint["indexed_companies"]:
            checkpoint["indexed_companies"].append(ticker)
        _save_checkpoint(checkpoint_path, checkpoint)

    client = get_qdrant_client()
    get_filings_collection()
    all_docs: list[dict] = []
    offset = None
    while True:
        records, offset = client.scroll(
            collection_name=config.QDRANT_COLLECTION_NAME,
            limit=100,
            with_payload=True,
            offset=offset,
        )
        for record in records:
            payload = dict(record.payload) if record.payload else {}
            text = payload.pop("text", "")
            all_docs.append({"text": text, "metadata": payload})
        if offset is None:
            break

    if all_docs:
        bm25_index.build_bm25_index(all_docs)
        logger.info("BM25 index rebuilt from %d total documents.", len(all_docs))

    logger.info("Ingestion complete.")
    return 0


def main() -> None:
    """CLI entry point: run ingestion and exit with the returned status code."""
    raise SystemExit(run_ingestion())


if __name__ == "__main__":
    main()
