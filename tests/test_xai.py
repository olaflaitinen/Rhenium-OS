# Copyright (c) 2025 Skolyn LLC. All rights reserved.

"""Tests for XAI module."""

import pytest
import numpy as np


class TestExplanationSchema:
    """Test XAI data structures."""

    def test_finding_creation(self):
        """Test Finding creation and serialization."""
        from rhenium.xai.explanation_schema import Finding, FindingSeverity

        finding = Finding(
            finding_type="lesion",
            description="Test lesion",
            confidence=0.95,
            severity=FindingSeverity.MODERATE,
        )

        assert finding.finding_id
        assert finding.confidence == 0.95

        data = finding.to_dict()
        assert data["finding_type"] == "lesion"
        assert data["severity"] == "moderate"

    def test_visual_evidence(self):
        """Test VisualEvidence creation."""
        from rhenium.xai.explanation_schema import VisualEvidence

        heatmap = np.random.rand(64, 64).astype(np.float32)
        evidence = VisualEvidence(
            artifact_type="heatmap",
            data=heatmap,
            description="Test heatmap",
        )

        assert evidence.evidence_id
        data = evidence.to_dict()
        assert data["artifact_type"] == "heatmap"


class TestEvidenceDossier:
    """Test Evidence Dossier."""

    def test_dossier_creation(self):
        """Test dossier creation."""
        from rhenium.xai.evidence_dossier import EvidenceDossier, create_dossier_for_finding
        from rhenium.xai.explanation_schema import Finding

        finding = Finding(finding_type="test", description="Test")
        dossier = create_dossier_for_finding(finding, study_uid="123")

        assert dossier.finding == finding
        assert dossier.study_uid == "123"

    def test_dossier_serialization(self, tmp_path):
        """Test dossier save/load."""
        from rhenium.xai.evidence_dossier import EvidenceDossier

        dossier = EvidenceDossier(study_uid="test_study")
        path = dossier.save(tmp_path / "dossier.json")

        assert path.exists()
