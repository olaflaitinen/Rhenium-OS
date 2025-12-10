# Copyright (c) 2025 Skolyn LLC. All rights reserved.
# SPDX-License-Identifier: EUPL-1.1

"""
Pipeline Runner
===============

Configuration-driven pipeline execution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from rhenium.core.logging import get_pipeline_logger
from rhenium.core.registry import registry, ComponentType
from rhenium.core.errors import PipelineError
from rhenium.pipelines.base_pipeline import BasePipeline, PipelineResult

logger = get_pipeline_logger()

CONFIGS_DIR = Path(__file__).parent / "configs"


class PipelineRunner:
    """
    Configuration-driven pipeline runner.

    Loads pipeline configurations and instantiates appropriate pipelines.
    """

    def __init__(self, config_name: str | None = None, config_path: Path | None = None):
        self.config_name = config_name
        self.config_path = config_path
        self.config: dict[str, Any] = {}
        self.pipeline: BasePipeline | None = None

        if config_name:
            self._load_config_by_name(config_name)
        elif config_path:
            self._load_config_from_path(config_path)

    def _load_config_by_name(self, name: str) -> None:
        """Load configuration by name."""
        config_file = CONFIGS_DIR / f"{name}.yaml"
        if not config_file.exists():
            raise PipelineError(f"Configuration '{name}' not found", pipeline_name=name)
        self._load_config_from_path(config_file)

    def _load_config_from_path(self, path: Path) -> None:
        """Load configuration from file."""
        logger.info("Loading pipeline configuration", path=str(path))
        with open(path) as f:
            self.config = yaml.safe_load(f)

    @classmethod
    def from_config(cls, config_name: str) -> "PipelineRunner":
        """Create runner from configuration name."""
        return cls(config_name=config_name)

    @classmethod
    def list_available_configs(cls) -> list[str]:
        """List available pipeline configurations."""
        return [p.stem for p in CONFIGS_DIR.glob("*.yaml")]

    def build_pipeline(self) -> BasePipeline:
        """Build pipeline from configuration."""
        if not self.config:
            raise PipelineError("No configuration loaded")

        pipeline_type = self.config.get("pipeline_type", "generic")

        try:
            pipeline_cls = registry.get_pipeline(pipeline_type)
            self.pipeline = pipeline_cls(config=self.config)
            return self.pipeline
        except Exception as e:
            logger.warning(f"Pipeline type '{pipeline_type}' not in registry, using generic")
            from rhenium.pipelines.base_pipeline import BasePipeline
            # Would need concrete implementation
            raise PipelineError(f"Cannot instantiate pipeline: {e}")

    def run(self, source: Any) -> PipelineResult:
        """Run the configured pipeline."""
        if self.pipeline is None:
            self.build_pipeline()

        if self.pipeline is None:
            raise PipelineError("Failed to build pipeline")

        return self.pipeline.run(source)


def load_pipeline_config(config_name: str) -> dict[str, Any]:
    """Load a pipeline configuration by name."""
    config_file = CONFIGS_DIR / f"{config_name}.yaml"
    if not config_file.exists():
        raise PipelineError(f"Configuration '{config_name}' not found")
    with open(config_file) as f:
        return yaml.safe_load(f)
