# Copyright (c) 2025 Skolyn LLC. All rights reserved.
# SPDX-License-Identifier: EUPL-1.1

"""
Rhenium OS Error Hierarchy
==========================

Defines the exception hierarchy for Rhenium OS. All custom exceptions inherit
from RheniumError, providing consistent error handling and categorization
across the system.

Error messages are designed to be:
    - Informative for debugging and troubleshooting
    - Safe for logging (no PHI leakage)
    - Suitable for regulatory review and audit

Usage:
    from rhenium.core.errors import DataIngestionError

    raise DataIngestionError(
        "Failed to parse DICOM file",
        source="dicom_io",
        details={"file_count": 10, "failed_count": 2}
    )
"""

from __future__ import annotations

from typing import Any


class RheniumError(Exception):
    """
    Base exception for all Rhenium OS errors.

    Attributes:
        message: Human-readable error description.
        source: Component or module that raised the error.
        details: Additional context (must not contain PHI).
        error_code: Optional error code for classification.
    """

    def __init__(
        self,
        message: str,
        source: str | None = None,
        details: dict[str, Any] | None = None,
        error_code: str | None = None,
    ) -> None:
        """
        Initialize a RheniumError.

        Args:
            message: Human-readable error description.
            source: Component or module that raised the error.
            details: Additional context (must not contain PHI).
            error_code: Optional error code for classification.
        """
        self.message = message
        self.source = source
        self.details = details or {}
        self.error_code = error_code
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format the full error message."""
        parts = [self.message]
        if self.source:
            parts.append(f"[source={self.source}]")
        if self.error_code:
            parts.append(f"[code={self.error_code}]")
        return " ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert error to dictionary for serialization.

        Returns:
            Dictionary representation of the error.
        """
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "source": self.source,
            "details": self.details,
            "error_code": self.error_code,
        }


class ConfigurationError(RheniumError):
    """
    Raised when configuration is invalid or missing.

    This includes errors in settings files, environment variables,
    or runtime configuration.
    """

    def __init__(
        self,
        message: str,
        setting_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        if setting_name:
            details["setting_name"] = setting_name
        super().__init__(message, source="config", details=details, **kwargs)


class DataIngestionError(RheniumError):
    """
    Raised when data ingestion fails.

    This covers errors during DICOM parsing, file format issues,
    metadata extraction failures, and data validation problems.
    """

    def __init__(
        self,
        message: str,
        file_path: str | None = None,
        format_type: str | None = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        if file_path:
            # Store only filename, not full path (to avoid PHI in paths)
            from pathlib import Path
            details["filename"] = Path(file_path).name
        if format_type:
            details["format"] = format_type
        super().__init__(message, source="data", details=details, **kwargs)


class ReconstructionError(RheniumError):
    """
    Raised when image reconstruction fails.

    This includes errors in the Rhenium Reconstruction Engine, k-space processing,
    sinogram reconstruction, and image enhancement pipelines.
    """

    def __init__(
        self,
        message: str,
        modality: str | None = None,
        pipeline_step: str | None = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        if modality:
            details["modality"] = modality
        if pipeline_step:
            details["pipeline_step"] = pipeline_step
        super().__init__(message, source="reconstruction", details=details, **kwargs)


class ModelInferenceError(RheniumError):
    """
    Raised when model inference fails.

    This covers errors in perception models (segmentation, detection,
    classification), including model loading, input validation,
    and inference execution.
    """

    def __init__(
        self,
        message: str,
        model_name: str | None = None,
        model_version: str | None = None,
        task: str | None = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        if model_name:
            details["model_name"] = model_name
        if model_version:
            details["model_version"] = model_version
        if task:
            details["task"] = task
        super().__init__(message, source="perception", details=details, **kwargs)


class MedGemmaError(RheniumError):
    """
    Raised when MedGemma integration fails.

    This includes connection errors, API failures, response parsing
    issues, and content filtering rejections.
    """

    def __init__(
        self,
        message: str,
        operation: str | None = None,
        status_code: int | None = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        if operation:
            details["operation"] = operation
        if status_code is not None:
            details["status_code"] = status_code
        super().__init__(message, source="medgemma", details=details, **kwargs)


class ExplainabilityError(RheniumError):
    """
    Raised when XAI artifact generation fails.

    This covers errors in generating visual explanations, computing
    quantitative metrics, or assembling evidence dossiers.
    """

    def __init__(
        self,
        message: str,
        explanation_type: str | None = None,
        finding_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        if explanation_type:
            details["explanation_type"] = explanation_type
        if finding_id:
            details["finding_id"] = finding_id
        super().__init__(message, source="xai", details=details, **kwargs)


class PipelineError(RheniumError):
    """
    Raised when pipeline execution fails.

    This covers errors in pipeline configuration, step execution,
    orchestration, and result assembly.
    """

    def __init__(
        self,
        message: str,
        pipeline_name: str | None = None,
        pipeline_version: str | None = None,
        step_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        if pipeline_name:
            details["pipeline_name"] = pipeline_name
        if pipeline_version:
            details["pipeline_version"] = pipeline_version
        if step_name:
            details["step_name"] = step_name
        super().__init__(message, source="pipeline", details=details, **kwargs)


class ValidationError(RheniumError):
    """
    Raised when input validation fails.

    This is used for validating user inputs, configuration values,
    and data format requirements.
    """

    def __init__(
        self,
        message: str,
        field_name: str | None = None,
        expected_type: str | None = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        if field_name:
            details["field_name"] = field_name
        if expected_type:
            details["expected_type"] = expected_type
        super().__init__(message, source="validation", details=details, **kwargs)


class RegistryError(RheniumError):
    """
    Raised when component registry operations fail.

    This includes errors in component registration, lookup, and
    version resolution.
    """

    def __init__(
        self,
        message: str,
        component_type: str | None = None,
        component_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        if component_type:
            details["component_type"] = component_type
        if component_name:
            details["component_name"] = component_name
        super().__init__(message, source="registry", details=details, **kwargs)


class GovernanceError(RheniumError):
    """
    Raised when governance operations fail.

    This covers errors in audit logging, model card generation,
    and compliance validation.
    """

    def __init__(
        self,
        message: str,
        operation: str | None = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        if operation:
            details["operation"] = operation
        super().__init__(message, source="governance", details=details, **kwargs)
