# Copyright (c) 2025 Skolyn LLC. All rights reserved.

"""
MedGemma Adapter
================

Abstract interface for MedGemma integration, supporting local and remote deployment.
Provides structured interfaces for report generation, explanation, consistency
validation, and multi-turn clinical dialogs.

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from rhenium.core.config import get_settings, MedGemmaBackend
from rhenium.core.errors import MedGemmaError
from rhenium.core.logging import get_medgemma_logger
from rhenium.xai.explanation_schema import Finding, NarrativeEvidence

logger = get_medgemma_logger()


@dataclass
class ReportDraft:
    """Draft radiology report from MedGemma."""
    indication: str = ""
    technique: str = ""
    comparison: str = ""
    findings: str = ""
    impression: str = ""
    recommendations: str = ""
    raw_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsistencyReport:
    """Consistency validation result."""
    is_consistent: bool = True
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)


class MedGemmaClient(ABC):
    """Abstract MedGemma client interface."""

    @abstractmethod
    def generate_report(
        self,
        findings: list[Finding],
        clinical_context: dict[str, Any] | None = None,
    ) -> ReportDraft:
        """Generate structured radiology report."""
        pass

    @abstractmethod
    def explain_finding(
        self,
        finding: Finding,
        context: dict[str, Any] | None = None,
    ) -> NarrativeEvidence:
        """Generate explanation for a specific finding."""
        pass

    @abstractmethod
    def validate_consistency(
        self,
        findings: list[Finding],
        metadata: dict[str, Any] | None = None,
    ) -> ConsistencyReport:
        """Validate consistency across findings."""
        pass

    @abstractmethod
    def answer_question(
        self,
        question: str,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Answer clinical question in context."""
        pass


class StubMedGemmaClient(MedGemmaClient):
    """Stub client for testing without MedGemma."""

    def generate_report(
        self,
        findings: list[Finding],
        clinical_context: dict[str, Any] | None = None,
    ) -> ReportDraft:
        logger.info("Generating report (stub)", num_findings=len(findings))
        findings_text = "\n".join([f"- {f.description}" for f in findings]) or "No significant findings."
        return ReportDraft(
            indication="Clinical indication as provided.",
            technique="Standard imaging protocol.",
            findings=findings_text,
            impression="AI-assisted analysis complete. Human review required.",
        )

    def explain_finding(
        self,
        finding: Finding,
        context: dict[str, Any] | None = None,
    ) -> NarrativeEvidence:
        logger.info("Generating explanation (stub)", finding_id=finding.finding_id)
        return NarrativeEvidence(
            explanation=f"Finding: {finding.description}. Confidence: {finding.confidence:.1%}. "
                       f"This finding was detected by automated analysis and requires human verification.",
            limitations=["This is an AI-generated explanation requiring radiologist review."],
            confidence_statement=f"Model confidence: {finding.confidence:.1%}",
        )

    def validate_consistency(
        self,
        findings: list[Finding],
        metadata: dict[str, Any] | None = None,
    ) -> ConsistencyReport:
        return ConsistencyReport(is_consistent=True)

    def answer_question(
        self,
        question: str,
        context: dict[str, Any] | None = None,
    ) -> str:
        return "This question requires human expert review."


class LocalMedGemmaClient(MedGemmaClient):
    """Client for local MedGemma deployment."""

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path
        self.model = None
        self._loaded = False

    def load(self) -> None:
        """Load MedGemma model weights."""
        logger.info("Loading local MedGemma model", path=self.model_path)
        # Placeholder for actual model loading
        self._loaded = True

    def generate_report(self, findings: list[Finding], clinical_context: dict[str, Any] | None = None) -> ReportDraft:
        if not self._loaded:
            self.load()
        # Placeholder - would call local model
        return StubMedGemmaClient().generate_report(findings, clinical_context)

    def explain_finding(self, finding: Finding, context: dict[str, Any] | None = None) -> NarrativeEvidence:
        if not self._loaded:
            self.load()
        return StubMedGemmaClient().explain_finding(finding, context)

    def validate_consistency(self, findings: list[Finding], metadata: dict[str, Any] | None = None) -> ConsistencyReport:
        return ConsistencyReport(is_consistent=True)

    def answer_question(self, question: str, context: dict[str, Any] | None = None) -> str:
        if not self._loaded:
            self.load()
        return StubMedGemmaClient().answer_question(question, context)


class RemoteMedGemmaClient(MedGemmaClient):
    """Client for remote MedGemma API."""

    def __init__(self, endpoint: str, timeout: int = 60):
        self.endpoint = endpoint
        self.timeout = timeout

    def generate_report(self, findings: list[Finding], clinical_context: dict[str, Any] | None = None) -> ReportDraft:
        logger.info("Calling remote MedGemma", endpoint=self.endpoint)
        # Placeholder - would make HTTP request
        return StubMedGemmaClient().generate_report(findings, clinical_context)

    def explain_finding(self, finding: Finding, context: dict[str, Any] | None = None) -> NarrativeEvidence:
        return StubMedGemmaClient().explain_finding(finding, context)

    def validate_consistency(self, findings: list[Finding], metadata: dict[str, Any] | None = None) -> ConsistencyReport:
        return ConsistencyReport(is_consistent=True)

    def answer_question(self, question: str, context: dict[str, Any] | None = None) -> str:
        return StubMedGemmaClient().answer_question(question, context)


def get_medgemma_client() -> MedGemmaClient:
    """Get MedGemma client based on settings."""
    settings = get_settings()

    if settings.medgemma_backend == MedGemmaBackend.STUB:
        logger.info("Using stub MedGemma client")
        return StubMedGemmaClient()
    elif settings.medgemma_backend == MedGemmaBackend.LOCAL:
        return LocalMedGemmaClient(str(settings.medgemma_model_path) if settings.medgemma_model_path else None)
    elif settings.medgemma_backend == MedGemmaBackend.REMOTE:
        return RemoteMedGemmaClient(settings.medgemma_endpoint, settings.medgemma_timeout)
    else:
        return StubMedGemmaClient()
