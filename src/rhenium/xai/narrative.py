"""Narrative generation for evidence dossiers using MedGemma."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
from rhenium.xai.evidence_dossier import Finding, EvidenceDossier, NarrativeEvidence


@dataclass
class NarrativeConfig:
    """Configuration for narrative generation."""
    include_limitations: bool = True
    confidence_threshold: float = 0.7
    max_length: int = 500
    language: str = "en"


class NarrativeGenerator:
    """Generate natural language narratives for findings."""

    def __init__(self, medgemma_client: Any = None, config: NarrativeConfig | None = None):
        self.client = medgemma_client
        self.config = config or NarrativeConfig()

    def generate(
        self,
        finding: Finding,
        image: np.ndarray | None = None,
        context: dict[str, Any] | None = None,
    ) -> NarrativeEvidence:
        """Generate narrative for a finding."""
        if self.client is not None:
            return self._generate_with_medgemma(finding, image, context)
        return self._generate_template(finding, context)

    def _generate_template(
        self,
        finding: Finding,
        context: dict[str, Any] | None = None,
    ) -> NarrativeEvidence:
        """Generate template-based narrative."""
        confidence_desc = self._confidence_to_text(finding.confidence)

        explanation = (
            f"AI analysis detected a {finding.finding_type} finding. "
            f"{finding.description} "
            f"The confidence level is {confidence_desc} ({finding.confidence:.0%}). "
        )

        if finding.location:
            explanation += f"Location: {finding.location}. "

        if finding.quantitative_evidence:
            measures = ", ".join(
                f"{q.evidence_type}: {q.value:.2f} {q.unit or ''}"
                for q in finding.quantitative_evidence
            )
            explanation += f"Measurements: {measures}. "

        limitations = [
            "This is an AI-generated finding and requires clinical verification.",
            "Performance may vary across different patient populations.",
            "Not validated for clinical decision-making.",
        ]

        return NarrativeEvidence(
            evidence_id=f"narr_{finding.finding_id}",
            explanation=explanation,
            limitations=limitations if self.config.include_limitations else [],
            confidence_statement=f"Model confidence: {finding.confidence:.0%}",
        )

    def _generate_with_medgemma(
        self,
        finding: Finding,
        image: np.ndarray | None,
        context: dict[str, Any] | None,
    ) -> NarrativeEvidence:
        """Generate narrative using MedGemma VLM."""
        prompt = self._build_prompt(finding, context)
        response = self.client.generate_narrative(
            image=image,
            findings=finding.to_dict(),
            prompt=prompt,
        )

        return NarrativeEvidence(
            evidence_id=f"narr_{finding.finding_id}",
            explanation=response,
            limitations=["AI-generated narrative - verify with clinical context"],
        )

    def _build_prompt(self, finding: Finding, context: dict[str, Any] | None) -> str:
        """Build prompt for MedGemma."""
        return (
            f"Generate a clinical narrative for the following AI finding:\n"
            f"Type: {finding.finding_type}\n"
            f"Description: {finding.description}\n"
            f"Confidence: {finding.confidence:.0%}\n"
            f"Location: {finding.location or 'Not specified'}\n"
        )

    def _confidence_to_text(self, confidence: float) -> str:
        if confidence >= 0.9:
            return "high"
        elif confidence >= 0.7:
            return "moderate"
        else:
            return "low"
