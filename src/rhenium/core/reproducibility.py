"""
Rhenium OS Reproducibility Utilities.

This module provides utilities for ensuring reproducible experiments
including seed setting, deterministic mode, and experiment tracking.
"""

from __future__ import annotations

import contextlib
import os
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Generator

import numpy as np

from rhenium.core.config import get_settings


def set_seed(seed: int | None = None) -> int:
    """
    Set random seeds for reproducibility.

    Sets seeds for Python random, NumPy, and PyTorch (if available).

    Args:
        seed: Random seed. Uses settings.seed if not provided.

    Returns:
        The seed that was set.
    """
    if seed is None:
        seed = get_settings().seed

    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch (if available)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    return seed


def set_deterministic(enabled: bool | None = None) -> None:
    """
    Enable or disable deterministic algorithms.

    Note: Deterministic mode may reduce performance.

    Args:
        enabled: Whether to enable. Uses settings.deterministic if not provided.
    """
    if enabled is None:
        enabled = get_settings().deterministic

    # Environment variable for cuBLAS
    if enabled:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # PyTorch deterministic settings
    try:
        import torch

        torch.use_deterministic_algorithms(enabled)
        if enabled:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        else:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
    except ImportError:
        pass


@contextlib.contextmanager
def get_deterministic_context(seed: int | None = None, enabled: bool = True) -> Generator[None, None, None]:
    """
    Context manager for temporary deterministic execution.
    
    Restores previous state upon exit.

    Args:
        seed: Seed to set within context
        enabled: Whether to enable determinism
    """
    # Save previous state
    # Note: capturing full RNG state is complex; this is a simplified restore
    # that mainly toggles settings back.
    prev_settings = get_settings()
    was_deterministic = prev_settings.deterministic
    
    try:
        if seed is not None:
             set_seed(seed)
        set_deterministic(enabled)
        yield
    finally:
        # Restore settings (seed restoration is skipped as it would require saving full state)
        set_deterministic(was_deterministic)


@dataclass
class ExperimentConfig:
    """
    Configuration snapshot for experiment reproducibility.

    Captures all relevant settings for reproducing an experiment.
    """

    seed: int
    deterministic: bool
    device: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Package versions
    rhenium_version: str = ""
    torch_version: str = ""
    monai_version: str = ""

    # Data configuration
    data_config: dict[str, Any] = field(default_factory=dict)

    # Model configuration
    model_config: dict[str, Any] = field(default_factory=dict)

    # Training configuration
    training_config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Capture package versions."""
        # Rhenium version
        try:
            from rhenium import __version__

            self.rhenium_version = __version__
        except ImportError:
            pass

        # PyTorch version
        try:
            import torch

            self.torch_version = torch.__version__
        except ImportError:
            pass

        # MONAI version
        try:
            import monai

            self.monai_version = monai.__version__
        except ImportError:
            pass

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "seed": self.seed,
            "deterministic": self.deterministic,
            "device": self.device,
            "timestamp": self.timestamp.isoformat(),
            "versions": {
                "rhenium": self.rhenium_version,
                "torch": self.torch_version,
                "monai": self.monai_version,
            },
            "data": self.data_config,
            "model": self.model_config,
            "training": self.training_config,
        }

    @classmethod
    def from_settings(cls) -> "ExperimentConfig":
        """Create config from current settings."""
        settings = get_settings()
        return cls(
            seed=settings.seed,
            deterministic=settings.deterministic,
            device=settings.get_effective_device(),
        )


def setup_reproducibility(seed: int | None = None, deterministic: bool | None = None) -> int:
    """
    Set up reproducibility with seed and deterministic mode.

    Args:
        seed: Random seed
        deterministic: Enable deterministic algorithms

    Returns:
        The seed that was set
    """
    seed = set_seed(seed)
    set_deterministic(deterministic)
    return seed
