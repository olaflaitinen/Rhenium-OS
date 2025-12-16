"""
Rhenium OS Error Taxonomy.

This module defines a hierarchical exception system for the platform.
All exceptions inherit from RheniumError for consistent error handling.
"""

from __future__ import annotations

from typing import Any


class RheniumError(Exception):
    """
    Base exception for all Rhenium OS errors.

    Attributes:
        message: Human-readable error message
        code: Machine-readable error code
        details: Additional error context
    """

    def __init__(
        self,
        message: str,
        code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code or self.__class__.__name__
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        """Serialize error for API responses."""
        return {
            "error": self.code,
            "message": self.message,
            "details": self.details,
        }


# Configuration Errors
class ConfigurationError(RheniumError):
    """Error in configuration or settings."""

    pass


class InvalidSettingsError(ConfigurationError):
    """Invalid configuration settings."""

    pass


class MissingConfigError(ConfigurationError):
    """Required configuration is missing."""

    pass


# Data Errors
class DataError(RheniumError):
    """Base class for data-related errors."""

    pass


class DataLoadError(DataError):
    """Error loading data from file or source."""

    pass


class DataFormatError(DataError):
    """Data format is invalid or unsupported."""

    pass


class DICOMError(DataError):
    """Error processing DICOM files."""

    pass


class NIfTIError(DataError):
    """Error processing NIfTI files."""

    pass


class MetadataError(DataError):
    """Error with data metadata."""

    pass


class DeidentificationError(DataError):
    """Error during data de-identification."""

    pass


# Model Errors
class ModelError(RheniumError):
    """Base class for model-related errors."""

    pass


class ModelLoadError(ModelError):
    """Error loading model weights or configuration."""

    pass


class ModelInferenceError(ModelError):
    """Error during model inference."""

    pass


class ModelTrainingError(ModelError):
    """Error during model training."""

    pass


class CheckpointError(ModelError):
    """Error with model checkpoints."""

    pass


# Pipeline Errors
class PipelineError(RheniumError):
    """Base class for pipeline-related errors."""

    pass


class PipelineConfigError(PipelineError):
    """Invalid pipeline configuration."""

    pass


class PipelineExecutionError(PipelineError):
    """Error during pipeline execution."""

    pass


class PipelineNotFoundError(PipelineError):
    """Requested pipeline not found in registry."""

    pass


# Validation Errors
class ValidationError(RheniumError):
    """Base class for validation errors."""

    pass


class InputValidationError(ValidationError):
    """Invalid input data or parameters."""

    pass


class OutputValidationError(ValidationError):
    """Output failed validation checks."""

    pass


class SchemaValidationError(ValidationError):
    """Data does not match expected schema."""

    pass


# Registry Errors
class RegistryError(RheniumError):
    """Base class for registry errors."""

    pass


class ComponentNotFoundError(RegistryError):
    """Component not found in registry."""

    pass


class ComponentAlreadyExistsError(RegistryError):
    """Component already registered."""

    pass


# Backend/Server Errors
class ServerError(RheniumError):
    """Base class for server errors."""

    pass


class JobNotFoundError(ServerError):
    """Job not found."""

    pass


class JobExecutionError(ServerError):
    """Error executing job."""

    pass


# MedGemma Errors
class MedGemmaError(RheniumError):
    """Base class for MedGemma integration errors."""

    pass


class MedGemmaConnectionError(MedGemmaError):
    """Error connecting to MedGemma backend."""

    pass


class MedGemmaDisabledError(MedGemmaError):
    """MedGemma is disabled in configuration."""

    pass


# Reconstruction Errors
class ReconstructionError(RheniumError):
    """Base class for reconstruction errors."""

    pass


class KSpaceError(ReconstructionError):
    """Error processing k-space data."""

    pass


class SinogramError(ReconstructionError):
    """Error processing sinogram data."""

    pass


# XAI Errors
class XAIError(RheniumError):
    """Base class for explainability errors."""

    pass


class EvidenceDossierError(XAIError):
    """Error generating Evidence Dossier."""

    pass


class SaliencyError(XAIError):
    """Error generating saliency maps."""

    pass
