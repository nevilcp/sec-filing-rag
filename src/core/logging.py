"""Structured logging configuration for the SEC compliance RAG system.

Responsibilities:
  - Emit single-line JSON log records for external aggregators.
  - Create isolated, daily-rotating log channels (app, api, rag,
    phoenix, system, audit).
  - Keep audit records off stdout.
"""

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
    """Serialize every log record to a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        """Return the record as a single-line JSON string.

        The record's 'extra' dict, if present, is merged into the top
        level, and exception details go under 'exception'.

        Args:
            record: The log record to serialize.

        Returns:
            The JSON text, with all non-ASCII characters escaped.
        """
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
    """Build a stdout handler with the JSON formatter attached."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(_JsonFormatter())
    return handler


def _make_rotating_file_handler(
    log_dir: Path,
    filename: str,
    level: int = logging.DEBUG,
) -> logging.handlers.TimedRotatingFileHandler:
    """Build a daily-rotating file handler that writes JSON logs.

    Args:
        log_dir: Directory where the log file is created.
        filename: Base filename without extension.
        level: Minimum log level this handler emits.
    """
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
    """Set up the stdout handler and every channel file handler.

    Sets the root logger to DEBUG so handlers control filtering, then
    attaches a stdout stream handler plus a rotating file handler for
    each channel (app, api, rag, phoenix, system, audit). The audit
    channel does not propagate, which keeps sensitive data off stdout.

    Args:
        level: Minimum log level emitted to stdout.
    """
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
    """Attach a rotating file handler to a named logger channel.

    Sets the channel logger to DEBUG so the handler alone decides what
    gets written.

    Args:
        channel: Fully-qualified logger name, for example
            'sec_rag.audit'.
        log_dir: Directory for the rotating log file.
        filename: Base filename, without extension, for the log file.
        file_level: Minimum log level written to the file.
        propagate: If False, records do not reach the root logger. Use
            False for channels that must not emit to stdout.
    """
    logger = logging.getLogger(channel)
    logger.setLevel(logging.DEBUG)
    logger.propagate = propagate
    logger.addHandler(_make_rotating_file_handler(log_dir, filename, file_level))


def get_logger(name: str) -> logging.Logger:
    """Return a logger for the given module name.

    Use it as ``logger = get_logger(__name__)`` at module level. These
    records reach the root stdout handler only. The channel file
    handlers attach to the sec_rag.* loggers from the get_*_logger
    functions below, so use those for records that must reach a file.

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        The standard library Logger for that name.
    """
    return logging.getLogger(name)


def get_app_logger() -> logging.Logger:
    """Return the app lifecycle channel logger (``sec_rag.app``)."""
    return logging.getLogger(CHANNEL_APP)


def get_api_logger() -> logging.Logger:
    """Return the HTTP access log channel logger (``sec_rag.api``)."""
    return logging.getLogger(CHANNEL_API)


def get_rag_logger() -> logging.Logger:
    """Return the RAG pipeline channel logger (``sec_rag.rag``)."""
    return logging.getLogger(CHANNEL_RAG)


def get_phoenix_logger() -> logging.Logger:
    """Return the Phoenix/OTEL channel logger (``sec_rag.phoenix``)."""
    return logging.getLogger(CHANNEL_PHOENIX)


def get_system_logger() -> logging.Logger:
    """Return the system channel logger (``sec_rag.system``)."""
    return logging.getLogger(CHANNEL_SYSTEM)


def get_audit_logger() -> logging.Logger:
    """Return the security audit channel logger (``sec_rag.audit``).

    Records on this channel do not propagate to stdout.
    """
    return logging.getLogger(CHANNEL_AUDIT)
