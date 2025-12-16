"""Benchmarking framework for model evaluation."""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
import json
import time
import numpy as np


@dataclass
class BenchmarkResult:
    """Single benchmark run result."""
    benchmark_name: str
    model_name: str
    metrics: dict[str, float]
    runtime_seconds: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark_name,
            "model": self.model_name,
            "metrics": self.metrics,
            "runtime_s": self.runtime_seconds,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""
    name: str
    dataset: str
    metrics: list[str]
    num_samples: int = -1  # -1 for all
    batch_size: int = 1
    device: str = "cuda"


class BenchmarkRunner:
    """Run standardized benchmarks."""

    def __init__(self, output_dir: Path | None = None):
        self.output_dir = output_dir or Path("./benchmark_results")
        self.results: list[BenchmarkResult] = []

    def run(
        self,
        config: BenchmarkConfig,
        model: Any,
        data_loader: Any,
        metric_fns: dict[str, Callable],
    ) -> BenchmarkResult:
        """Run a benchmark."""
        start_time = time.time()
        predictions = []
        targets = []

        for batch in data_loader:
            pred = model(batch["input"])
            predictions.append(pred)
            targets.append(batch["target"])

        runtime = time.time() - start_time

        # Compute metrics
        predictions = np.concatenate(predictions)
        targets = np.concatenate(targets)

        metrics = {}
        for name, fn in metric_fns.items():
            if name in config.metrics:
                metrics[name] = fn(predictions, targets)

        result = BenchmarkResult(
            benchmark_name=config.name,
            model_name=str(type(model).__name__),
            metrics=metrics,
            runtime_seconds=runtime,
            metadata={
                "dataset": config.dataset,
                "num_samples": len(predictions),
                "device": config.device,
            },
        )

        self.results.append(result)
        return result

    def save_results(self, filename: str = "results.json") -> Path:
        """Save all results to file."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / filename

        with open(output_path, "w") as f:
            json.dump([r.to_dict() for r in self.results], f, indent=2)

        return output_path

    def summary(self) -> dict[str, Any]:
        """Generate summary statistics."""
        if not self.results:
            return {}

        metrics_by_model = {}
        for r in self.results:
            if r.model_name not in metrics_by_model:
                metrics_by_model[r.model_name] = []
            metrics_by_model[r.model_name].append(r.metrics)

        summary = {}
        for model, metrics_list in metrics_by_model.items():
            summary[model] = {
                k: np.mean([m[k] for m in metrics_list if k in m])
                for k in metrics_list[0].keys()
            }

        return summary
