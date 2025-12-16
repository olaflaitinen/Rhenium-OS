"""
Rhenium OS Structured Logging.

This module provides structured logging using structlog with
support for JSON and console output formats.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog
from structlog.types import Processor

from rhenium.core.config import LogFormat, LogLevel, get_settings


def configure_logging(
    level: LogLevel | None = None,
    format: LogFormat | None = None,
) -> None:
    """
    Configure the logging system.

    Args:
        level: Log level (uses settings if not specified)
        format: Output format (uses settings if not specified)
    """
    settings = get_settings()
    level = level or settings.log_level
    format = format or settings.log_format

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.value),
    )

    # Build processor chain
    processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    if format == LogFormat.JSON:
        processors.append(structlog.processors.JSONRenderer())
    elif format == LogFormat.CONSOLE:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))
    else:  # PLAIN
        processors.append(
            structlog.processors.KeyValueRenderer(
                key_order=["timestamp", "level", "event"]
            )
        )

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.value)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None, **initial_context: Any) -> structlog.BoundLogger:
    """
    Get a structured logger instance.

    Args:
        name: Logger name (module name recommended)
        **initial_context: Initial context to bind to logger

    Returns:
        Bound logger instance

    Example:
        logger = get_logger(__name__)
        logger.info("Processing started", study_uid="1.2.3")
    """
    logger = structlog.get_logger(name)
    if initial_context:
        logger = logger.bind(**initial_context)
    return logger


class LogContext:
    """
    Context manager for adding temporary logging context.

    Example:
        with LogContext(pipeline="segmentation", study="123"):
            logger.info("Starting")  # Includes pipeline and study
    """

    def __init__(self, **context: Any) -> None:
        self.context = context
        self._token: Any = None

    def __enter__(self) -> "LogContext":
        self._token = structlog.contextvars.bind_contextvars(**self.context)
        return self

    def __exit__(self, *args: Any) -> None:
        structlog.contextvars.unbind_contextvars(*self.context.keys())


def log_exception(
    logger: structlog.BoundLogger,
    exc: Exception,
    message: str = "Exception occurred",
    **context: Any,
) -> None:
    """
    Log an exception with full context.

    Args:
        logger: Logger instance
        exc: Exception to log
        message: Log message
        **context: Additional context
    """
    logger.exception(
        message,
        exc_type=type(exc).__name__,
        exc_message=str(exc),
        **context,
    )


# Auto-configure on import if settings available
try:
    configure_logging()
except Exception:
    pass  # Settings not available yet
