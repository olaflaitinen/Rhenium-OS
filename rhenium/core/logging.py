# Copyright (c) 2025 Skolyn LLC. All rights reserved.

"""
Rhenium OS Logging System
=========================

Structured logging framework for Rhenium OS using structlog. Provides
consistent, configurable logging across all subsystems with support for
JSON and text output formats.

The logging system creates separate loggers for each subsystem to enable
fine-grained control over logging output:

    - rhenium.core: Core infrastructure logging
    - rhenium.data: Data ingestion and preprocessing
    - rhenium.reconstruction: Reconstruction pipeline logging
    - rhenium.perception: Perception model inference
    - rhenium.xai: Explainability artifact generation
    - rhenium.medgemma: MedGemma integration
    - rhenium.pipeline: Pipeline orchestration
    - rhenium.governance: Audit and governance logging

Usage:
    from rhenium.core.logging import get_logger

    logger = get_logger(__name__)
    logger.info("Processing study", study_id="12345", modality="MRI")
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog
from structlog.typing import Processor


if TYPE_CHECKING:
    from rhenium.core.config import RheniumSettings


# Module-level flag to track initialization
_logging_configured = False


def _add_timestamp(
    logger: logging.Logger,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Add ISO-8601 timestamp to log events."""
    event_dict["timestamp"] = datetime.now(timezone.utc).isoformat()
    return event_dict


def _add_logger_name(
    logger: logging.Logger,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Add logger name to log events."""
    record = event_dict.get("_record")
    if record:
        event_dict["logger"] = record.name
    return event_dict


def _sanitize_phi(
    logger: logging.Logger,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """
    Sanitize potential PHI from log events.

    This processor removes or masks fields that may contain Protected
    Health Information (PHI) to comply with GDPR and HIPAA requirements.
    """
    phi_fields = {
        "patient_name",
        "patient_id",
        "patient_birth_date",
        "patient_address",
        "ssn",
        "mrn",
        "accession_number",
    }

    for field in phi_fields:
        if field in event_dict:
            event_dict[field] = "[REDACTED]"

    return event_dict


def _add_context(
    logger: logging.Logger,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Add Rhenium OS context to log events."""
    event_dict["service"] = "rhenium-os"
    return event_dict


def configure_logging(
    level: str = "INFO",
    log_format: str = "text",
    log_to_file: bool = False,
    logs_dir: Path | None = None,
) -> None:
    """
    Configure the global logging system.

    This function sets up structlog with the appropriate processors and
    handlers based on the configuration. It should be called once at
    application startup.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_format: Output format ('json' or 'text').
        log_to_file: Whether to write logs to files.
        logs_dir: Directory for log files (required if log_to_file is True).
    """
    global _logging_configured

    if _logging_configured:
        return

    # Convert string level to logging constant
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    # Build processor chain
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        _add_timestamp,
        _add_context,
        _sanitize_phi,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    # Select renderer based on format
    if log_format == "json":
        renderer: Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(
            colors=sys.stdout.isatty(),
            exception_formatter=structlog.dev.plain_traceback,
        )

    # Add format-specific processors
    processors: list[Processor] = [
        *shared_processors,
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
    ]

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Create formatter for stdlib handlers
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    # Configure root handler
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    root_logger.addHandler(console_handler)

    # Add file handler if requested
    if log_to_file and logs_dir:
        logs_dir = Path(logs_dir)
        logs_dir.mkdir(parents=True, exist_ok=True)

        log_file = logs_dir / f"rhenium_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        root_logger.addHandler(file_handler)

    # Set levels for third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    _logging_configured = True


def configure_logging_from_settings(settings: RheniumSettings) -> None:
    """
    Configure logging from RheniumSettings.

    Args:
        settings: The Rhenium OS settings instance.
    """
    configure_logging(
        level=settings.log_level.value,
        log_format=settings.log_format,
        log_to_file=settings.log_to_file,
        logs_dir=settings.logs_dir,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """
    Get a logger instance for the specified name.

    If logging has not been configured, it will be configured with
    default settings.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        A bound structlog logger instance.
    """
    if not _logging_configured:
        configure_logging()

    return structlog.get_logger(name)


# Convenience loggers for common subsystems
def get_core_logger() -> structlog.stdlib.BoundLogger:
    """Get logger for core infrastructure."""
    return get_logger("rhenium.core")


def get_data_logger() -> structlog.stdlib.BoundLogger:
    """Get logger for data ingestion and preprocessing."""
    return get_logger("rhenium.data")


def get_reconstruction_logger() -> structlog.stdlib.BoundLogger:
    """Get logger for reconstruction pipelines."""
    return get_logger("rhenium.reconstruction")


def get_perception_logger() -> structlog.stdlib.BoundLogger:
    """Get logger for perception model inference."""
    return get_logger("rhenium.perception")


def get_xai_logger() -> structlog.stdlib.BoundLogger:
    """Get logger for XAI artifact generation."""
    return get_logger("rhenium.xai")


def get_medgemma_logger() -> structlog.stdlib.BoundLogger:
    """Get logger for MedGemma integration."""
    return get_logger("rhenium.medgemma")


def get_pipeline_logger() -> structlog.stdlib.BoundLogger:
    """Get logger for pipeline orchestration."""
    return get_logger("rhenium.pipeline")


def get_governance_logger() -> structlog.stdlib.BoundLogger:
    """Get logger for audit and governance."""
    return get_logger("rhenium.governance")
