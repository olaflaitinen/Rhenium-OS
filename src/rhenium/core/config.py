"""
Rhenium OS Configuration System.

This module provides Pydantic-based configuration with environment variable support.
All settings can be overridden via RHENIUM_ prefixed environment variables.

Example:
    export RHENIUM_DATA_DIR=/path/to/data
    export RHENIUM_DEVICE=cuda
"""

from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DeviceType(str, Enum):
    """Supported compute device types."""

    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    AUTO = "auto"


class LogLevel(str, Enum):
    """Logging verbosity levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(str, Enum):
    """Logging output formats."""

    JSON = "json"
    CONSOLE = "console"
    PLAIN = "plain"


class MedGemmaBackend(str, Enum):
    """MedGemma client backend options."""

    STUB = "stub"
    LOCAL = "local"
    REMOTE = "remote"
    DISABLED = "disabled"


class RheniumSettings(BaseSettings):
    """
    Central configuration for Rhenium OS.

    All settings can be overridden via environment variables with the
    RHENIUM_ prefix. For example, RHENIUM_DATA_DIR=/path/to/data.

    Attributes:
        data_dir: Base directory for all input and output data.
        models_dir: Base directory for storing trained models.
        logs_dir: Base directory for logging output.
        cache_dir: Base directory for temporary caches.
        device: Default compute device to use.
        cuda_device_id: Specific CUDA device index.
        num_workers: Number of dataloader workers.
        log_level: Logging verbosity level.
        log_format: Logging output format.
        seed: Random seed for reproducibility.
        deterministic: Enable deterministic algorithms.
        medgemma_enabled: Whether MedGemma integration is enabled.
        medgemma_backend: MedGemma backend to use.
        medgemma_endpoint: Remote MedGemma API endpoint.
    """

    model_config = SettingsConfigDict(
        env_prefix="RHENIUM_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Path settings
    data_dir: Path = Field(
        default=Path("./data"),
        description="Base directory for input/output data",
    )
    models_dir: Path = Field(
        default=Path("./models"),
        description="Base directory for trained models",
    )
    logs_dir: Path = Field(
        default=Path("./logs"),
        description="Base directory for logs",
    )
    cache_dir: Path = Field(
        default=Path("./cache"),
        description="Base directory for caches",
    )

    # Compute settings
    device: DeviceType = Field(
        default=DeviceType.AUTO,
        description="Compute device (cpu, cuda, mps, auto)",
    )
    cuda_device_id: int = Field(
        default=0,
        ge=0,
        description="CUDA device index",
    )
    num_workers: int = Field(
        default=4,
        ge=0,
        description="Number of dataloader workers",
    )

    # Logging settings
    log_level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Logging verbosity level",
    )
    log_format: LogFormat = Field(
        default=LogFormat.CONSOLE,
        description="Logging output format",
    )

    # Reproducibility settings
    seed: int = Field(
        default=42,
        description="Random seed for reproducibility",
    )
    deterministic: bool = Field(
        default=False,
        description="Enable deterministic algorithms (may reduce performance)",
    )

    # MedGemma settings
    medgemma_enabled: bool = Field(
        default=False,
        description="Enable MedGemma integration",
    )
    medgemma_backend: MedGemmaBackend = Field(
        default=MedGemmaBackend.STUB,
        description="MedGemma backend (stub, local, remote, disabled)",
    )
    medgemma_endpoint: str = Field(
        default="",
        description="Remote MedGemma API endpoint URL",
    )
    medgemma_model_path: Path | None = Field(
        default=None,
        description="Path to local MedGemma model weights",
    )

    @field_validator("data_dir", "models_dir", "logs_dir", "cache_dir", mode="before")
    @classmethod
    def resolve_path(cls, v: str | Path) -> Path:
        """Resolve path to absolute and expand user."""
        return Path(v).expanduser().resolve()

    @model_validator(mode="after")
    def validate_cuda_config(self) -> "RheniumSettings":
        """Validate CUDA configuration if CUDA device is selected."""
        if self.device == DeviceType.CUDA:
            try:
                import torch

                if not torch.cuda.is_available():
                    raise ValueError("CUDA device requested but CUDA is not available")
                if self.cuda_device_id >= torch.cuda.device_count():
                    raise ValueError(
                        f"CUDA device {self.cuda_device_id} not found. "
                        f"Available devices: 0-{torch.cuda.device_count() - 1}"
                    )
            except ImportError:
                pass  # PyTorch not installed yet
        return self

    def ensure_directories(self) -> None:
        """Create all configured directories if they don't exist."""
        for path in [self.data_dir, self.models_dir, self.logs_dir, self.cache_dir]:
            path.mkdir(parents=True, exist_ok=True)

    def get_effective_device(self) -> str:
        """Get the effective device string for PyTorch."""
        if self.device == DeviceType.AUTO:
            try:
                import torch

                if torch.cuda.is_available():
                    return f"cuda:{self.cuda_device_id}"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    return "mps"
                else:
                    return "cpu"
            except ImportError:
                return "cpu"
        elif self.device == DeviceType.CUDA:
            return f"cuda:{self.cuda_device_id}"
        else:
            return self.device.value


@lru_cache(maxsize=1)
def get_settings() -> RheniumSettings:
    """
    Get cached settings instance.

    Returns:
        RheniumSettings: The global settings instance.
    """
    return RheniumSettings()


def clear_settings_cache() -> None:
    """Clear the settings cache. Useful for testing."""
    get_settings.cache_clear()
