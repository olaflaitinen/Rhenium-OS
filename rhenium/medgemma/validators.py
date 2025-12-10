# Copyright (c) 2025 Skolyn LLC. All rights reserved.

"""
MedGemma Validators
===================

Safety checks and validation logic for MedGemma outputs. These validators
ensure consistency, plausibility, and flag high-risk outputs for mandatory
human review.

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from rhenium.xai.explanation_schema import Finding
from rhenium.medgemma.adapter import ReportDraft
from rhenium.core.logging import get_medgemma_logger

logger = get_medgemma_logger()


@dataclass
class ValidationResult:
    """Result of a validation check."""
    passed: bool
    validator_name: str
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    requires_human_review: bool = False


class BaseValidator(ABC):
    """Abstract base class for MedGemma output validators."""
    
    name: str = "base_validator"
    
    @abstractmethod
    def validate(self, **kwargs: Any) -> ValidationResult:
        """Perform validation."""
        pass


class NarrativeQuantitativeConsistencyValidator(BaseValidator):
    """
    Validates that narrative descriptions are consistent with quantitative data.
    
    Detects potential hallucinations where MedGemma generates text that
    contradicts measured values (e.g., "small lesion" when measurement > 5cm).
    """
    
    name = "narrative_quantitative_consistency"
    
    SIZE_DESCRIPTORS = {
        "tiny": (0, 5),        # mm
        "small": (0, 15),      # mm
        "moderate": (10, 40),  # mm
        "large": (30, 100),    # mm
        "massive": (80, 500),  # mm
    }
    
    def validate(
        self,
        narrative: str,
        findings: list[Finding] | None = None,
        measurements: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> ValidationResult:
        """
        Check narrative against quantitative measurements.
        
        Args:
            narrative: MedGemma-generated narrative text
            findings: List of findings with measurements
            measurements: Direct measurement dictionary
            
        Returns:
            ValidationResult with any inconsistencies noted
        """
        issues = []
        warnings = []
        
        narrative_lower = narrative.lower()
        
        # Extract any measurements from findings
        all_measurements = measurements.copy() if measurements else {}
        if findings:
            for finding in findings:
                if hasattr(finding, 'measurements') and finding.measurements:
                    all_measurements.update(finding.measurements)
        
        # Check size descriptor consistency
        for descriptor, (min_mm, max_mm) in self.SIZE_DESCRIPTORS.items():
            if descriptor in narrative_lower:
                # Check if any measurement contradicts this descriptor
                for name, value in all_measurements.items():
                    if "size" in name.lower() or "diameter" in name.lower():
                        if value < min_mm or value > max_mm:
                            issues.append(
                                f"Narrative describes '{descriptor}' but measured {name}={value}mm "
                                f"(expected {min_mm}-{max_mm}mm)"
                            )
        
        passed = len(issues) == 0
        
        logger.debug("Narrative consistency check",
                    passed=passed,
                    num_issues=len(issues))
        
        return ValidationResult(
            passed=passed,
            validator_name=self.name,
            issues=issues,
            warnings=warnings,
            requires_human_review=len(issues) > 0,
        )


class LateralityConsistencyValidator(BaseValidator):
    """
    Validates laterality consistency in findings and narratives.
    
    Ensures left/right designations do not contradict each other within
    a single report or finding set.
    """
    
    name = "laterality_consistency"
    
    def validate(
        self,
        narrative: str = "",
        findings: list[Finding] | None = None,
        **kwargs: Any,
    ) -> ValidationResult:
        """Check laterality consistency."""
        issues = []
        warnings = []
        findings = findings or []
        
        # Track laterality from findings
        lateralities_mentioned = set()
        for finding in findings:
            if finding.laterality:
                lateralities_mentioned.add(finding.laterality.lower())
        
        # Check for contradictions
        if "left" in lateralities_mentioned and "right" in lateralities_mentioned:
            if "bilateral" not in lateralities_mentioned:
                warnings.append(
                    "Both left and right findings present without bilateral designation"
                )
        
        # Check narrative for contradictions
        narrative_lower = narrative.lower()
        if "left" in narrative_lower and "right" in narrative_lower:
            if "bilateral" not in narrative_lower and len(lateralities_mentioned) == 1:
                issues.append(
                    "Narrative mentions both left and right but findings are unilateral"
                )
        
        passed = len(issues) == 0
        
        return ValidationResult(
            passed=passed,
            validator_name=self.name,
            issues=issues,
            warnings=warnings,
            requires_human_review=len(issues) > 0,
        )


class HighRiskFindingValidator(BaseValidator):
    """
    Flags high-risk findings that require mandatory human review.
    
    Certain clinical situations (suspected malignancy, urgent findings)
    should always be reviewed by a radiologist regardless of AI confidence.
    """
    
    name = "high_risk_finding"
    
    HIGH_RISK_TERMS = [
        "malignant", "malignancy", "cancer", "carcinoma",
        "metastasis", "metastatic", "tumor", "tumour",
        "hemorrhage", "bleeding", "stroke", "infarct",
        "fracture", "pneumothorax", "emergent", "urgent",
        "life-threatening", "critical",
    ]
    
    def validate(
        self,
        narrative: str = "",
        findings: list[Finding] | None = None,
        report: ReportDraft | None = None,
        **kwargs: Any,
    ) -> ValidationResult:
        """Check for high-risk findings requiring human review."""
        issues = []
        requires_review = False
        
        # Combine all text for checking
        all_text = narrative.lower()
        if report:
            all_text += " " + report.findings.lower()
            all_text += " " + report.impression.lower()
        
        # Check for high-risk terms
        found_terms = []
        for term in self.HIGH_RISK_TERMS:
            if term in all_text:
                found_terms.append(term)
                requires_review = True
        
        if found_terms:
            logger.info("High-risk terms detected",
                       terms=found_terms,
                       requires_review=True)
        
        return ValidationResult(
            passed=True,  # Not a failure, just a flag
            validator_name=self.name,
            issues=[],
            warnings=[f"High-risk term detected: {t}" for t in found_terms],
            requires_human_review=requires_review,
        )


class OverconfidenceValidator(BaseValidator):
    """
    Detects potentially overconfident language in AI outputs.
    
    MedGemma should express appropriate uncertainty; definitive
    language without supporting evidence should be flagged.
    """
    
    name = "overconfidence_check"
    
    OVERCONFIDENT_PHRASES = [
        "definitely", "certainly", "absolutely",
        "100%", "guaranteed", "undoubtedly",
        "there is no possibility", "impossible",
    ]
    
    APPROPRIATE_UNCERTAINTY = [
        "likely", "probably", "may represent",
        "differential includes", "cannot exclude",
        "recommend", "suggest", "consider",
    ]
    
    def validate(
        self,
        narrative: str = "",
        report: ReportDraft | None = None,
        **kwargs: Any,
    ) -> ValidationResult:
        """Check for overconfident language."""
        warnings = []
        
        all_text = narrative.lower()
        if report:
            all_text += " " + report.impression.lower()
        
        # Check for overconfident phrases
        for phrase in self.OVERCONFIDENT_PHRASES:
            if phrase in all_text:
                warnings.append(f"Potentially overconfident language: '{phrase}'")
        
        # Check for complete lack of uncertainty language
        has_uncertainty = any(u in all_text for u in self.APPROPRIATE_UNCERTAINTY)
        if not has_uncertainty and len(all_text) > 100:
            warnings.append("No uncertainty language detected in substantial output")
        
        return ValidationResult(
            passed=True,
            validator_name=self.name,
            issues=[],
            warnings=warnings,
            requires_human_review=len(warnings) > 2,
        )


@dataclass
class ValidationSuite:
    """Runs multiple validators on MedGemma outputs."""
    
    validators: list[BaseValidator] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.validators:
            self.validators = [
                NarrativeQuantitativeConsistencyValidator(),
                LateralityConsistencyValidator(),
                HighRiskFindingValidator(),
                OverconfidenceValidator(),
            ]
    
    def validate_all(
        self,
        narrative: str = "",
        findings: list[Finding] | None = None,
        report: ReportDraft | None = None,
        measurements: dict[str, float] | None = None,
    ) -> list[ValidationResult]:
        """Run all validators."""
        results = []
        
        for validator in self.validators:
            result = validator.validate(
                narrative=narrative,
                findings=findings,
                report=report,
                measurements=measurements,
            )
            results.append(result)
        
        return results
    
    def requires_human_review(self, results: list[ValidationResult]) -> bool:
        """Check if any result requires human review."""
        return any(r.requires_human_review for r in results)
    
    def get_all_issues(self, results: list[ValidationResult]) -> list[str]:
        """Collect all issues from results."""
        issues = []
        for result in results:
            issues.extend(result.issues)
        return issues


def get_default_validation_suite() -> ValidationSuite:
    """Get the default validation suite."""
    return ValidationSuite()
