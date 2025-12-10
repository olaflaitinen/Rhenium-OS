# Copyright (c) 2025 Skolyn LLC. All rights reserved.
# SPDX-License-Identifier: EUPL-1.1

"""Tests for MedGemma adapter."""

import pytest


class TestMedGemmaAdapter:
    """Test MedGemma client implementations."""

    def test_stub_client(self):
        """Test stub client functionality."""
        from rhenium.medgemma.adapter import StubMedGemmaClient
        from rhenium.xai.explanation_schema import Finding

        client = StubMedGemmaClient()

        findings = [Finding(finding_type="lesion", description="Test")]
        report = client.generate_report(findings)

        assert report.findings
        assert "human review" in report.impression.lower() or "review" in report.impression.lower()

    def test_explain_finding(self):
        """Test finding explanation."""
        from rhenium.medgemma.adapter import StubMedGemmaClient
        from rhenium.xai.explanation_schema import Finding

        client = StubMedGemmaClient()
        finding = Finding(finding_type="test", description="Test finding", confidence=0.8)

        explanation = client.explain_finding(finding)
        assert explanation.explanation
        assert explanation.limitations

    def test_client_factory(self, mock_settings):
        """Test client factory function."""
        from rhenium.medgemma.adapter import get_medgemma_client, StubMedGemmaClient

        client = get_medgemma_client()
        assert isinstance(client, StubMedGemmaClient)
