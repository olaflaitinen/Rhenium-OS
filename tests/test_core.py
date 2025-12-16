"""Core module tests."""

import pytest
from rhenium.core.config import RheniumSettings, get_settings
from rhenium.core.registry import Registry, ComponentType
from rhenium.core.errors import RheniumError, ConfigurationError


class TestSettings:
    def test_default_settings(self):
        settings = RheniumSettings()
        assert settings.device is not None
        assert settings.seed == 42

    def test_get_settings_cached(self):
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2


class TestRegistry:
    def test_register_and_get(self):
        registry = Registry()

        @registry.register("model", "test_model", version="1.0.0")
        class TestModel:
            pass

        cls = registry.get("model", "test_model")
        assert cls is TestModel

    def test_get_nonexistent(self):
        registry = Registry()
        with pytest.raises(KeyError):
            registry.get("model", "nonexistent")

    def test_list_components(self):
        registry = Registry()

        @registry.register(ComponentType.MODEL, "m1")
        class M1:
            pass

        components = registry.list_components(ComponentType.MODEL)
        assert len(components) == 1


class TestErrors:
    def test_error_to_dict(self):
        err = RheniumError("Test error", code="TEST_CODE")
        d = err.to_dict()
        assert d["error"] == "TEST_CODE"
        assert d["message"] == "Test error"

    def test_inheritance(self):
        err = ConfigurationError("Config error")
        assert isinstance(err, RheniumError)
