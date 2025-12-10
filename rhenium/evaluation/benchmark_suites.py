# Copyright (c) 2025 Skolyn LLC. All rights reserved.

"""Benchmark suites for evaluation."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from rhenium.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    suite_name: str
    metrics: dict[str, float] = field(default_factory=dict)
    per_sample_results: list[dict] = field(default_factory=list)


@dataclass
class BenchmarkSuite:
    """Configurable benchmark suite."""
    name: str
    pipeline_config: str
    metrics: list[str] = field(default_factory=list)
    dataset_path: str = ""

    def run(self) -> BenchmarkResult:
        """Execute benchmark suite."""
        logger.info("Running benchmark suite", name=self.name)
        return BenchmarkResult(suite_name=self.name)


KNEE_MRI_BENCHMARK = BenchmarkSuite(
    name="knee_mri_benchmark",
    pipeline_config="mri_knee_default",
    metrics=["dice", "hausdorff", "sensitivity", "specificity"],
)

BRAIN_LESION_BENCHMARK = BenchmarkSuite(
    name="brain_lesion_benchmark",
    pipeline_config="brain_lesion_default",
    metrics=["dice", "lesion_count_accuracy", "volume_error"],
)
