# Copyright (c) 2025 Skolyn LLC. All rights reserved.

"""Tests for pipeline framework."""

import pytest


class TestPipelineRunner:
    """Test pipeline runner."""

    def test_list_configs(self):
        """Test listing available configurations."""
        from rhenium.pipelines.pipeline_runner import PipelineRunner

        configs = PipelineRunner.list_available_configs()
        assert "mri_knee_default" in configs
        assert "brain_lesion_default" in configs

    def test_load_config(self):
        """Test loading configuration."""
        from rhenium.pipelines.pipeline_runner import load_pipeline_config

        config = load_pipeline_config("mri_knee_default")
        assert config["name"] == "mri_knee_default"
        assert "perception" in config


class TestPipelineResult:
    """Test pipeline result structure."""

    def test_result_creation(self):
        """Test PipelineResult creation."""
        from rhenium.pipelines.base_pipeline import PipelineResult

        result = PipelineResult(pipeline_name="test", pipeline_version="1.0.0")
        assert result.run_id
        assert result.status == "success"

    def test_result_serialization(self):
        """Test result serialization."""
        from rhenium.pipelines.base_pipeline import PipelineResult

        result = PipelineResult(pipeline_name="test", pipeline_version="1.0.0")
        data = result.to_dict()

        assert data["pipeline_name"] == "test"
        assert "started_at" in data
