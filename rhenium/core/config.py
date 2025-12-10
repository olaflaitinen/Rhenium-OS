# Copyright (c) 2025 Skolyn LLC. All rights reserved.

"""
Rhenium OS Configuration System
===============================

Pydantic-based configuration management for Rhenium OS. Provides centralized
settings for paths, hardware, integration endpoints, logging, and evaluation
parameters. Supports environment variable overrides and profile-based loading.

Usage:
    from rhenium.core.config import get_settings

    settings = get_settings()
    print(settings.data_dir)
    print(settings.device)
"""

from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DeviceType(str, Enum):
    """Supported compute device types."""

    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    AUTO = "auto"


class LogLevel(str, Enum):
    """Logging levels supported by Rhenium OS."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class MedGemmaBackend(str, Enum):
    """MedGemma deployment backend options."""

    LOCAL = "local"
    REMOTE = "remote"
    STUB = "stub"


class RheniumSettings(BaseSettings):
    """
    Central configuration for Rhenium OS.

    All settings can be overridden via environment variables with the
    RHENIUM_ prefix. For example, RHENIUM_DATA_DIR=/path/to/data.

    Attributes:
        data_dir: Root directory for data storage.
        models_dir: Directory containing trained model weights.
        logs_dir: Directory for log files.
        cache_dir: Directory for cached data and intermediate results.
        results_dir: Default directory for pipeline outputs.
        device: Compute device selection (cpu, cuda, mps, auto).
        cuda_device_ids: List of CUDA device IDs to use when device is cuda.
        num_workers: Number of worker threads for data loading.
        batch_size: Default batch size for inference.
        log_level: Logging verbosity level.
        log_format: Log output format (json or text).
        log_to_file: Whether to write logs to files.
        medgemma_backend: MedGemma deployment backend.
        medgemma_endpoint: URL for remote MedGemma endpoint.
        medgemma_model_path: Path to local MedGemma model weights.
        medgemma_timeout: Timeout in seconds for MedGemma requests.
        enable_audit_logging: Whether to enable audit logging.
        audit_log_dir: Directory for audit logs.
        enable_xai: Whether to generate XAI artifacts by default.
        confidence_threshold: Minimum confidence for findings.
        profile: Configuration profile name.
    """

    model_config = SettingsConfigDict(
        env_prefix="RHENIUM_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Path Configuration
    data_dir: Path = Field(
        default=Path("./data"),
        description="Root directory for data storage",
    )
    models_dir: Path = Field(
        default=Path("./models"),
        description="Directory containing trained model weights",
    )
    logs_dir: Path = Field(
        default=Path("./logs"),
        description="Directory for log files",
    )
    cache_dir: Path = Field(
        default=Path("./cache"),
        description="Directory for cached data and intermediate results",
    )
    results_dir: Path = Field(
        default=Path("./results"),
        description="Default directory for pipeline outputs",
    )

    # Hardware Configuration
    device: DeviceType = Field(
        default=DeviceType.AUTO,
        description="Compute device selection",
    )
    cuda_device_ids: list[int] = Field(
        default_factory=lambda: [0],
        description="List of CUDA device IDs to use",
    )
    num_workers: int = Field(
        default=4,
        ge=0,
        le=32,
        description="Number of worker threads for data loading",
    )
    batch_size: int = Field(
        default=1,
        ge=1,
        le=128,
        description="Default batch size for inference",
    )

    # Logging Configuration
    log_level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Logging verbosity level",
    )
    log_format: Literal["json", "text"] = Field(
        default="text",
        description="Log output format",
    )
    log_to_file: bool = Field(
        default=True,
        description="Whether to write logs to files",
    )

    # MedGemma Configuration
    medgemma_backend: MedGemmaBackend = Field(
        default=MedGemmaBackend.STUB,
        description="MedGemma deployment backend",
    )
    medgemma_endpoint: str = Field(
        default="http://localhost:8080/v1",
        description="URL for remote MedGemma endpoint",
    )
    medgemma_model_path: Path | None = Field(
        default=None,
        description="Path to local MedGemma model weights",
    )
    medgemma_timeout: int = Field(
        default=60,
        ge=1,
        le=300,
        description="Timeout in seconds for MedGemma requests",
    )
    medgemma_max_tokens: int = Field(
        default=4096,
        ge=256,
        le=32768,
        description="Maximum tokens for MedGemma generation",
    )

    # Governance Configuration
    enable_audit_logging: bool = Field(
        default=True,
        description="Whether to enable audit logging",
    )
    audit_log_dir: Path = Field(
        default=Path("./audit_logs"),
        description="Directory for audit logs",
    )

    # Evaluation Configuration
    enable_xai: bool = Field(
        default=True,
        description="Whether to generate XAI artifacts by default",
    )
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for findings",
    )

    # Profile Configuration
    profile: str = Field(
        default="default",
        description="Configuration profile name",
    )

    @field_validator("data_dir", "models_dir", "logs_dir", "cache_dir", "results_dir", "audit_log_dir", mode="before")
    @classmethod
    def expand_path(cls, v: Any) -> Path:
        """Expand user home directory and environment variables in paths."""
        if isinstance(v, str):
            v = os.path.expanduser(os.path.expandvars(v))
        return Path(v)

    @model_validator(mode="after")
    def validate_cuda_config(self) -> "RheniumSettings":
        """Validate CUDA configuration when using CUDA device."""
        if self.device == DeviceType.CUDA:
            if not self.cuda_device_ids:
                self.cuda_device_ids = [0]
        return self

    def get_device_string(self) -> str:
        """
        Get the device string for PyTorch.

        Returns:
            Device string (e.g., 'cuda:0', 'cpu', 'mps').
        """
        if self.device == DeviceType.AUTO:
            try:
                import torch
                if torch.cuda.is_available():
                    return f"cuda:{self.cuda_device_ids[0]}"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    return "mps"
                else:
                    return "cpu"
            except ImportError:
                return "cpu"
        elif self.device == DeviceType.CUDA:
            return f"cuda:{self.cuda_device_ids[0]}"
        else:
            return self.device.value

    def ensure_directories(self) -> None:
        """Create all configured directories if they do not exist."""
        for dir_path in [
            self.data_dir,
            self.models_dir,
            self.logs_dir,
            self.cache_dir,
            self.results_dir,
            self.audit_log_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> RheniumSettings:
    """
    Get the global Rhenium OS settings instance.

    Settings are loaded once and cached. To reload settings, call
    clear_settings_cache() first.

    Returns:
        RheniumSettings: The global settings instance.
    """
    return RheniumSettings()


def clear_settings_cache() -> None:
    """Clear the cached settings instance to force reload."""
    get_settings.cache_clear()


def load_profile_settings(profile: str) -> RheniumSettings:
    """
    Load settings for a specific profile.

    Profiles are loaded from .env.{profile} files if they exist.

    Args:
        profile: The profile name to load.

    Returns:
        RheniumSettings: Settings for the specified profile.
    """
    env_file = f".env.{profile}"
    if Path(env_file).exists():
        return RheniumSettings(_env_file=env_file, profile=profile)
    return RheniumSettings(profile=profile)
