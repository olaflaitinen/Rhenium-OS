"""MedGemma output validators for safety and quality."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import re


@dataclass
class ValidationResult:
    """Result of validation check."""
    is_valid: bool
    errors: list[str]
    warnings: list[str]


class ResponseValidator:
    """Validate MedGemma outputs for safety and quality."""

    def __init__(self):
        self._forbidden_patterns = [
            r'\b(diagnosis|diagnose|diagnosed)\b',
            r'\b(prescribe|prescription|medication)\b',
            r'\b(treatment plan)\b',
        ]

    def validate(self, response: str) -> ValidationResult:
        """Validate a MedGemma response."""
        errors = []
        warnings = []

        # Check for forbidden clinical assertions
        for pattern in self._forbidden_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                warnings.append(f"Response contains clinical terminology: {pattern}")

        # Check for required disclaimers
        if "ai" not in response.lower() and "automated" not in response.lower():
            warnings.append("Response should acknowledge AI-generated nature")

        # Check response length
        if len(response) < 50:
            warnings.append("Response may be too short for clinical context")
        if len(response) > 2000:
            warnings.append("Response may be too long - consider summarizing")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )


class FindingValidator:
    """Validate AI findings for consistency."""

    def validate_confidence(self, confidence: float) -> ValidationResult:
        """Validate confidence score."""
        errors = []
        warnings = []

        if confidence < 0 or confidence > 1:
            errors.append(f"Confidence {confidence} out of range [0, 1]")

        if confidence > 0.99:
            warnings.append("Very high confidence - verify calibration")

        return ValidationResult(len(errors) == 0, errors, warnings)

    def validate_measurements(
        self,
        measurements: dict[str, float],
        reference_ranges: dict[str, tuple[float, float]] | None = None,
    ) -> ValidationResult:
        """Validate quantitative measurements."""
        errors = []
        warnings = []

        for name, value in measurements.items():
            if value < 0:
                errors.append(f"Negative measurement: {name} = {value}")

            if reference_ranges and name in reference_ranges:
                low, high = reference_ranges[name]
                if value < low or value > high:
                    warnings.append(f"{name} = {value} outside reference range [{low}, {high}]")

        return ValidationResult(len(errors) == 0, errors, warnings)
