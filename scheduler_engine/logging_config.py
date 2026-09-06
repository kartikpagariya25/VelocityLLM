"""
VelocityLLM - Structured Logging & Correlation ID Tracking
Provides contextvars-based correlation ID propagation and JSON/structured log formatting.
"""

from contextvars import ContextVar
import datetime
import json
import logging
import sys
from typing import Any, Dict, Optional

# Global contextvar for request correlation ID
correlation_id_ctx: ContextVar[str] = ContextVar("correlation_id", default="")


def get_correlation_id() -> str:
    """Retrieve the current request correlation ID from context."""
    return correlation_id_ctx.get()


def set_correlation_id(corr_id: str) -> None:
    """Set the request correlation ID for the current async task context."""
    correlation_id_ctx.set(corr_id)


class CorrelationIdFilter(logging.Filter):
    """Logging filter that injects the active correlation_id into all log records."""
    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = get_correlation_id() or "-"
        return True


class StructuredJsonFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects for log aggregators (e.g. Datadog, ELK)."""
    def format(self, record: logging.LogRecord) -> str:
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.datetime.fromtimestamp(
                record.created, tz=datetime.timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "correlation_id": getattr(record, "correlation_id", "-"),
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        }
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)


def configure_logging(level: int = logging.INFO, json_format: bool = False) -> None:
    """Configures the root / velocityllm logger hierarchy."""
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(CorrelationIdFilter())

    if json_format:
        handler.setFormatter(StructuredJsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] [cid=%(correlation_id)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

    root_logger.addHandler(handler)
