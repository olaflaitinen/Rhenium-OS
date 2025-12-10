# Copyright (c) 2025 Skolyn LLC. All rights reserved.

"""
Rhenium OS Core Module
======================

Core infrastructure components including configuration, logging, error handling,
and component registry. These modules provide the foundational services used
throughout Rhenium OS.
"""

from rhenium.core.config import get_settings, RheniumSettings
from rhenium.core.errors import (
    RheniumError,
    ConfigurationError,
    DataIngestionError,
    ReconstructionError,
    ModelInferenceError,
    MedGemmaError,
    ExplainabilityError,
    PipelineError,
    ValidationError,
)
from rhenium.core.logging import get_logger, configure_logging
from rhenium.core.registry import registry, ComponentRegistry


__all__ = [
    # Configuration
    "get_settings",
    "RheniumSettings",
    # Errors
    "RheniumError",
    "ConfigurationError",
    "DataIngestionError",
    "ReconstructionError",
    "ModelInferenceError",
    "MedGemmaError",
    "ExplainabilityError",
    "PipelineError",
    "ValidationError",
    # Logging
    "get_logger",
    "configure_logging",
    # Registry
    "registry",
    "ComponentRegistry",
]
