# Copyright (c) 2025 Skolyn LLC. All rights reserved.
# SPDX-License-Identifier: EUPL-1.1

"""
Evidence Dossier
================

Central container for all evidence supporting a clinical finding.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from rhenium.xai.explanation_schema import (
    Finding,
    VisualEvidence,
    QuantitativeEvidence,
    NarrativeEvidence,
)
from rhenium.core.logging import get_xai_logger

logger = get_xai_logger()


@dataclass
class EvidenceDossier:
    """
    Complete evidence package for a clinical finding.

    The Evidence Dossier aggregates visual, quantitative, and narrative
    evidence to provide comprehensive explainability for each finding.
    """
    dossier_id: str = field(default_factory=lambda: uuid4().hex[:12])
    finding: Finding | None = None
    study_uid: str = ""
    series_uid: str = ""
    pipeline_name: str = ""
    pipeline_version: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def visual_evidence(self) -> list[VisualEvidence]:
        return self.finding.visual_evidence if self.finding else []

    @property
    def quantitative_evidence(self) -> list[QuantitativeEvidence]:
        return self.finding.quantitative_evidence if self.finding else []

    @property
    def narrative_evidence(self) -> list[NarrativeEvidence]:
        return self.finding.narrative_evidence if self.finding else []

    @property
    def summary(self) -> str:
        """Generate human-readable summary."""
        if not self.finding:
            return "No finding associated"

        parts = [f"Finding: {self.finding.description}"]
        parts.append(f"Confidence: {self.finding.confidence:.1%}")
        parts.append(f"Severity: {self.finding.severity.value}")

        if self.narrative_evidence:
            parts.append(f"Explanation: {self.narrative_evidence[0].explanation[:200]}...")

        return " | ".join(parts)

    def add_visual(self, evidence: VisualEvidence) -> None:
        """Add visual evidence."""
        if self.finding:
            self.finding.visual_evidence.append(evidence)

    def add_quantitative(self, evidence: QuantitativeEvidence) -> None:
        """Add quantitative evidence."""
        if self.finding:
            self.finding.quantitative_evidence.append(evidence)

    def add_narrative(self, evidence: NarrativeEvidence) -> None:
        """Add narrative evidence."""
        if self.finding:
            self.finding.narrative_evidence.append(evidence)

    def to_dict(self) -> dict[str, Any]:
        """Serialize dossier to dictionary."""
        return {
            "dossier_id": self.dossier_id,
            "finding": self.finding.to_dict() if self.finding else None,
            "study_uid": self.study_uid,
            "series_uid": self.series_uid,
            "pipeline_name": self.pipeline_name,
            "pipeline_version": self.pipeline_version,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save(self, path: str | Path) -> Path:
        """Save dossier to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())
        logger.info("Evidence dossier saved", path=str(path))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "EvidenceDossier":
        """Load dossier from JSON file."""
        path = Path(path)
        data = json.loads(path.read_text())
        # Would need full deserialization logic
        return cls(
            dossier_id=data.get("dossier_id", ""),
            study_uid=data.get("study_uid", ""),
            series_uid=data.get("series_uid", ""),
        )


def create_dossier_for_finding(
    finding: Finding,
    study_uid: str = "",
    series_uid: str = "",
    pipeline_name: str = "",
    pipeline_version: str = "",
) -> EvidenceDossier:
    """Create an evidence dossier for a finding."""
    return EvidenceDossier(
        finding=finding,
        study_uid=study_uid,
        series_uid=series_uid,
        pipeline_name=pipeline_name,
        pipeline_version=pipeline_version,
    )
