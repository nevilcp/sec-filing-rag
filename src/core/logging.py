"""Structured logging configuration for the SEC compliance RAG system."""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.core import config

_APP_ROOT = "sec_rag"
CHANNEL_APP = f"{_APP_ROOT}.app"
CHANNEL_API = f"{_APP_ROOT}.api"
CHANNEL_RAG = f"{_APP_ROOT}.rag"
CHANNEL_PHOENIX = f"{_APP_ROOT}.phoenix"
CHANNEL_SYSTEM = f"{_APP_ROOT}.system"
CHANNEL_AUDIT = f"{_APP_ROOT}.audit"


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        """Serialize a log record to JSON, merging extra fields and exception details."""
        parts = record.name.split(".")
        channel = parts[1] if len(parts) > 1 else parts[0]

        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "channel": channel,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if hasattr(record, "extra") and isinstance(record.extra, dict):
            payload.update(record.extra)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=True, default=str)


def _make_stream_handler(level: int = logging.DEBUG) -> logging.StreamHandler:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(_JsonFormatter())
    return handler


def _make_rotating_file_handler(
    log_dir: Path,
    filename: str,
    level: int = logging.DEBUG,
) -> logging.handlers.TimedRotatingFileHandler:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{filename}.log"

    handler = logging.handlers.TimedRotatingFileHandler(
        filename=str(log_path),
        when=config.LOG_ROTATION_WHEN,
        backupCount=config.LOG_ROTATION_BACKUP_COUNT,
        encoding="utf-8",
        utc=True,
    )
    handler.suffix = "%Y-%m-%d"
    handler.setLevel(level)
    handler.setFormatter(_JsonFormatter())
    return handler


def configure_logging(level: int = logging.INFO) -> None:
    """Configure JSON log channels and stdout, keeping audit records in their own file."""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()
    root.addHandler(_make_stream_handler(level))

    _attach_file_handler(
        CHANNEL_APP,
        config.LOG_DIR_APP,
        "app",
        file_level=logging.INFO,
    )

    _attach_file_handler(
        CHANNEL_API,
        config.LOG_DIR_API,
        "access",
        file_level=logging.INFO,
    )

    _attach_file_handler(
        CHANNEL_RAG,
        config.LOG_DIR_RAG,
        "pipeline",
        file_level=logging.DEBUG,
    )

    _attach_file_handler(
        CHANNEL_PHOENIX,
        config.LOG_DIR_PHOENIX,
        "phoenix",
        file_level=logging.DEBUG,
    )

    _attach_file_handler(
        CHANNEL_SYSTEM,
        config.LOG_DIR_SYSTEM,
        "system",
        file_level=logging.DEBUG,
    )

    _attach_file_handler(
        CHANNEL_AUDIT,
        config.LOG_DIR_AUDIT,
        "audit",
        file_level=logging.INFO,
        propagate=False,
    )

    for noisy in (
        "httpx",
        "httpcore",
        "urllib3",
        "requests",
        "uvicorn.access",
        "sqlalchemy.engine",
        "langchain_core",
        "opentelemetry",
        "grpc",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _attach_file_handler(
    channel: str,
    log_dir: Path,
    filename: str,
    file_level: int,
    *,
    propagate: bool = True,
) -> None:
    logger = logging.getLogger(channel)
    logger.setLevel(logging.DEBUG)
    logger.propagate = propagate
    logger.addHandler(_make_rotating_file_handler(log_dir, filename, file_level))


def get_logger(name: str) -> logging.Logger:
    """Map a source module to its logging channel."""
    return logging.getLogger(name)


def get_app_logger() -> logging.Logger:
    return logging.getLogger(CHANNEL_APP)


def get_api_logger() -> logging.Logger:
    return logging.getLogger(CHANNEL_API)


def get_rag_logger() -> logging.Logger:
    return logging.getLogger(CHANNEL_RAG)


def get_phoenix_logger() -> logging.Logger:
    return logging.getLogger(CHANNEL_PHOENIX)


def get_system_logger() -> logging.Logger:
    return logging.getLogger(CHANNEL_SYSTEM)


def get_audit_logger() -> logging.Logger:
    return logging.getLogger(CHANNEL_AUDIT)
