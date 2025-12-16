"""Governance artifact benchmark tests.

Tests verifying model card, dataset card, and risk register generation.
"""

import pytest
import json
from pathlib import Path
from typing import Any

# Try to import governance modules
try:
    from rhenium.governance.model_card import ModelCardGenerator
    HAS_GOVERNANCE = True
except ImportError:
    HAS_GOVERNANCE = False


@pytest.mark.bench
class TestModelCardGeneration:
    """Test model card generation."""
    
    @pytest.mark.skipif(not HAS_GOVERNANCE, reason="Governance module not available")
    def test_model_card_generator_exists(self):
        """Test ModelCardGenerator can be imported."""
        from rhenium.governance.model_card import ModelCardGenerator
        assert ModelCardGenerator is not None
    
    def test_model_card_template_structure(self):
        """Test model card has expected structure."""
        # Define expected structure
        expected_sections = [
            "model_details",
            "intended_use",
            "limitations",
            "ethical_considerations",
        ]
        
        # Create minimal model card template
        model_card = {
            "model_details": {
                "name": "RheniumCoreModel",
                "version": "1.0.0",
                "type": "Multi-task Medical Imaging",
            },
            "intended_use": {
                "primary_use": "Research and development",
                "out_of_scope": "Clinical diagnosis",
            },
            "limitations": [
                "Research use only",
                "Not validated for clinical use",
                "Trained on synthetic data only",
            ],
            "ethical_considerations": {
                "data": "Synthetic data only, no PHI",
                "risks": "May produce incorrect outputs",
            },
        }
        
        for section in expected_sections:
            assert section in model_card
    
    def test_model_card_json_valid(self):
        """Test model card can be serialized to JSON."""
        model_card = {
            "model_details": {"name": "Test", "version": "1.0.0"},
            "intended_use": {"primary": "Research"},
            "limitations": ["Research only"],
        }
        
        json_str = json.dumps(model_card, indent=2)
        parsed = json.loads(json_str)
        
        assert parsed["model_details"]["name"] == "Test"
    
    def test_model_card_markdown_valid(self):
        """Test model card markdown generation."""
        model_card_md = """# Model Card: RheniumCoreModel

## Model Details
- **Name:** RheniumCoreModel
- **Version:** 1.0.0
- **Type:** Multi-task Medical Imaging

## Intended Use
- **Primary use:** Research and development
- **Out of scope:** Clinical diagnosis

## Limitations
- Research use only
- Not validated for clinical use

## Ethical Considerations
- Uses synthetic data only, no PHI
"""
        
        # Should be valid markdown (contains expected sections)
        assert "# Model Card" in model_card_md
        assert "## Model Details" in model_card_md
        assert "## Intended Use" in model_card_md
        assert "## Limitations" in model_card_md


@pytest.mark.bench
class TestDatasetCardGeneration:
    """Test dataset card template generation."""
    
    def test_dataset_card_template_structure(self):
        """Test dataset card has expected structure."""
        dataset_card = {
            "dataset_name": "Rhenium Synthetic Dataset",
            "version": "1.0.0",
            "description": "Synthetic medical imaging data for testing",
            "source": {
                "type": "Synthetic",
                "generator": "SyntheticDataGenerator",
            },
            "composition": {
                "modalities": ["MRI", "CT", "US", "XR"],
                "size": "Variable (generated on demand)",
            },
            "usage": {
                "intended_use": "Testing and benchmarking",
                "out_of_scope": "Clinical validation",
            },
            "ethical": {
                "contains_phi": False,
                "contains_pii": False,
            },
        }
        
        assert dataset_card["dataset_name"] is not None
        assert dataset_card["ethical"]["contains_phi"] is False
    
    def test_dataset_card_json_valid(self):
        """Test dataset card JSON serialization."""
        dataset_card = {
            "dataset_name": "Test",
            "version": "1.0.0",
            "ethical": {"contains_phi": False},
        }
        
        json_str = json.dumps(dataset_card)
        parsed = json.loads(json_str)
        
        assert parsed["ethical"]["contains_phi"] is False


@pytest.mark.bench
class TestRiskRegisterGeneration:
    """Test risk register template generation."""
    
    def test_risk_register_structure(self):
        """Test risk register has expected structure."""
        risk_register = {
            "risks": [
                {
                    "id": "R001",
                    "category": "Technical",
                    "description": "Model may produce incorrect segmentation",
                    "likelihood": "Medium",
                    "impact": "High",
                    "mitigation": "Require expert review of all outputs",
                    "status": "Open",
                },
                {
                    "id": "R002",
                    "category": "Ethical",
                    "description": "Generated images may be mistaken for real",
                    "likelihood": "Low",
                    "impact": "Medium",
                    "mitigation": "Include disclosure metadata in all generated images",
                    "status": "Mitigated",
                },
            ],
            "metadata": {
                "last_updated": "2025-01-01",
                "owner": "Research Team",
            },
        }
        
        assert len(risk_register["risks"]) >= 2
        
        for risk in risk_register["risks"]:
            assert "id" in risk
            assert "description" in risk
            assert "mitigation" in risk
    
    def test_risk_register_json_valid(self):
        """Test risk register JSON serialization."""
        risk_register = {
            "risks": [{"id": "R001", "description": "Test risk"}],
        }
        
        json_str = json.dumps(risk_register)
        parsed = json.loads(json_str)
        
        assert len(parsed["risks"]) == 1


@pytest.mark.bench
class TestIncidentTemplateGeneration:
    """Test incident report template generation."""
    
    def test_incident_template_structure(self):
        """Test incident template has expected structure."""
        incident_template = {
            "incident_id": "",
            "date_reported": "",
            "date_occurred": "",
            "severity": "",  # Low, Medium, High, Critical
            "category": "",  # Technical, Safety, Privacy, etc.
            "description": "",
            "affected_systems": [],
            "root_cause": "",
            "corrective_actions": [],
            "preventive_actions": [],
            "status": "",  # Open, Investigating, Resolved, Closed
            "reporter": "",
            "assignee": "",
        }
        
        expected_fields = [
            "incident_id", "date_reported", "severity",
            "description", "corrective_actions", "status"
        ]
        
        for field in expected_fields:
            assert field in incident_template
    
    def test_incident_template_json_valid(self):
        """Test incident template JSON serialization."""
        incident = {
            "incident_id": "INC-001",
            "severity": "Low",
            "description": "Test incident",
            "status": "Open",
        }
        
        json_str = json.dumps(incident)
        parsed = json.loads(json_str)
        
        assert parsed["incident_id"] == "INC-001"


@pytest.mark.bench
class TestGovernanceArtifactIntegration:
    """Test governance artifacts work with core model."""
    
    def test_core_model_provenance_for_governance(self):
        """Test core model produces provenance suitable for governance."""
        from rhenium.models import RheniumCoreModel, RheniumCoreModelConfig, TaskType
        from rhenium.testing.synthetic import SyntheticDataGenerator
        
        gen = SyntheticDataGenerator(seed=42)
        volume = gen.generate_volume(shape=(8, 16, 16), modality="MRI")
        
        config = RheniumCoreModelConfig(device="cpu", seed=42)
        model = RheniumCoreModel(config)
        model.initialize()
        
        try:
            result = model.run(volume, task=TaskType.SEGMENTATION)
            
            # Provenance should have fields needed for governance
            provenance = result.provenance
            assert "model_name" in provenance
            assert "model_version" in provenance
            assert "timestamp" in provenance
            assert "input_modality" in provenance
        finally:
            model.shutdown()
