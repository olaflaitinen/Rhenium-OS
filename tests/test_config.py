# Copyright (c) 2025 Skolyn LLC. All rights reserved.
# SPDX-License-Identifier: EUPL-1.1

"""Tests for configuration system."""

import pytest
from rhenium.core.config import RheniumSettings, get_settings, DeviceType


class TestConfiguration:
    """Test configuration loading and validation."""

    def test_default_settings(self):
        """Test default settings are valid."""
        settings = RheniumSettings()
        assert settings.device == DeviceType.AUTO
        assert settings.batch_size >= 1
        assert settings.num_workers >= 0

    def test_environment_override(self, monkeypatch):
        """Test environment variable overrides."""
        from rhenium.core.config import clear_settings_cache

        monkeypatch.setenv("RHENIUM_BATCH_SIZE", "8")
        monkeypatch.setenv("RHENIUM_LOG_LEVEL", "DEBUG")
        clear_settings_cache()

        settings = RheniumSettings()
        assert settings.batch_size == 8
        assert settings.log_level.value == "DEBUG"

        clear_settings_cache()

    def test_device_string(self):
        """Test device string generation."""
        settings = RheniumSettings(device=DeviceType.CPU)
        assert settings.get_device_string() == "cpu"

    def test_path_expansion(self):
        """Test path expansion."""
        settings = RheniumSettings(data_dir="~/data")
        assert "~" not in str(settings.data_dir)
