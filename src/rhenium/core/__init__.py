"""Core module for Rhenium OS configuration, registry, and utilities."""

from rhenium.core.config import RheniumSettings, get_settings
from rhenium.core.registry import registry, get_registry, ComponentType
from rhenium.core.errors import (
    RheniumError,
    ConfigurationError,
    DataError,
    ModelError,
    PipelineError,
    ValidationError,
)
from rhenium.core.logging import get_logger, configure_logging

__all__ = [
    # Config
    "RheniumSettings",
    "get_settings",
    # Registry
    "registry",
    "get_registry",
    "ComponentType",
    # Errors
    "RheniumError",
    "ConfigurationError",
    "DataError",
    "ModelError",
    "PipelineError",
    "ValidationError",
    # Logging
    "get_logger",
    "configure_logging",
]
