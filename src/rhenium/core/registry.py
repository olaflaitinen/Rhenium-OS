"""
Rhenium OS Component Registry.

This module provides a versioned component registry for managing
pipelines, models, preprocessors, and other pluggable components.

Example:
    @registry.register("model", "unet3d", version="1.0.0")
    class UNet3D(nn.Module):
        pass

    model_cls = registry.get("model", "unet3d")
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, TypeVar

T = TypeVar("T")


class ComponentType(str, Enum):
    """Types of registrable components."""

    MODEL = "model"
    PIPELINE = "pipeline"
    PREPROCESSOR = "preprocessor"
    RECONSTRUCTOR = "reconstructor"
    GENERATOR = "generator"
    DISCRIMINATOR = "discriminator"
    ENCODER = "encoder"
    METRIC = "metric"
    LOSS = "loss"
    XAI = "xai"
    PROMPT = "prompt"
    DATASET = "dataset"


@dataclass
class ComponentInfo:
    """Metadata about a registered component."""

    name: str
    component_type: ComponentType
    version: str
    cls: type
    description: str = ""
    tags: list[str] = field(default_factory=list)
    registered_at: datetime = field(default_factory=datetime.utcnow)
    deprecated: bool = False
    deprecation_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "type": self.component_type.value,
            "version": self.version,
            "description": self.description,
            "tags": self.tags,
            "deprecated": self.deprecated,
        }


class Registry:
    """
    Thread-safe component registry with versioning support.

    The registry allows registering and retrieving components by type
    and name, with optional version specification.
    """

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._components: dict[str, dict[str, ComponentInfo]] = {}
        self._lock = threading.RLock()

    def _make_key(self, component_type: str | ComponentType, name: str) -> str:
        """Create unique key for component."""
        if isinstance(component_type, ComponentType):
            component_type = component_type.value
        return f"{component_type}:{name}"

    def register(
        self,
        component_type: str | ComponentType,
        name: str,
        version: str = "1.0.0",
        description: str = "",
        tags: list[str] | None = None,
    ) -> Callable[[type[T]], type[T]]:
        """
        Decorator to register a component.

        Args:
            component_type: Type of component (model, pipeline, etc.)
            name: Unique name for the component
            version: Semantic version string
            description: Human-readable description
            tags: Optional tags for categorization

        Returns:
            Decorator function that registers the class

        Example:
            @registry.register("model", "unet3d", version="1.0.0")
            class UNet3D(nn.Module):
                pass
        """

        def decorator(cls: type[T]) -> type[T]:
            self.register_class(
                cls=cls,
                component_type=component_type,
                name=name,
                version=version,
                description=description or cls.__doc__ or "",
                tags=tags or [],
            )
            return cls

        return decorator

    def register_class(
        self,
        cls: type,
        component_type: str | ComponentType,
        name: str,
        version: str = "1.0.0",
        description: str = "",
        tags: list[str] | None = None,
    ) -> None:
        """
        Register a class directly (non-decorator version).

        Args:
            cls: The class to register
            component_type: Type of component
            name: Unique name
            version: Semantic version
            description: Description
            tags: Optional tags
        """
        if isinstance(component_type, str):
            component_type = ComponentType(component_type)

        key = self._make_key(component_type, name)
        info = ComponentInfo(
            name=name,
            component_type=component_type,
            version=version,
            cls=cls,
            description=description,
            tags=tags or [],
        )

        with self._lock:
            if key not in self._components:
                self._components[key] = {}
            self._components[key][version] = info

    def get(
        self,
        component_type: str | ComponentType,
        name: str,
        version: str | None = None,
    ) -> type:
        """
        Get a registered component class.

        Args:
            component_type: Type of component
            name: Component name
            version: Specific version (None = latest)

        Returns:
            The registered class

        Raises:
            KeyError: If component not found
        """
        key = self._make_key(component_type, name)

        with self._lock:
            if key not in self._components:
                raise KeyError(f"Component not found: {component_type}:{name}")

            versions = self._components[key]
            if version is None:
                # Get latest version (sort by version string)
                latest_version = sorted(versions.keys())[-1]
                return versions[latest_version].cls
            elif version in versions:
                return versions[version].cls
            else:
                raise KeyError(f"Version {version} not found for {component_type}:{name}")

    def get_info(
        self,
        component_type: str | ComponentType,
        name: str,
        version: str | None = None,
    ) -> ComponentInfo:
        """Get component metadata."""
        key = self._make_key(component_type, name)

        with self._lock:
            if key not in self._components:
                raise KeyError(f"Component not found: {component_type}:{name}")

            versions = self._components[key]
            if version is None:
                latest_version = sorted(versions.keys())[-1]
                return versions[latest_version]
            elif version in versions:
                return versions[version]
            else:
                raise KeyError(f"Version {version} not found for {component_type}:{name}")

    def list_components(
        self,
        component_type: str | ComponentType | None = None,
        tags: list[str] | None = None,
    ) -> list[ComponentInfo]:
        """
        List registered components.

        Args:
            component_type: Filter by type (None = all)
            tags: Filter by tags (None = all)

        Returns:
            List of component metadata
        """
        results = []

        with self._lock:
            for key, versions in self._components.items():
                for info in versions.values():
                    # Filter by type
                    if component_type is not None:
                        if isinstance(component_type, str):
                            component_type = ComponentType(component_type)
                        if info.component_type != component_type:
                            continue

                    # Filter by tags
                    if tags is not None:
                        if not any(tag in info.tags for tag in tags):
                            continue

                    results.append(info)

        return results

    def list_pipelines(self) -> list[dict[str, Any]]:
        """List all registered pipelines."""
        return [
            info.to_dict()
            for info in self.list_components(ComponentType.PIPELINE)
        ]

    def list_models(self) -> list[dict[str, Any]]:
        """List all registered models."""
        return [
            info.to_dict()
            for info in self.list_components(ComponentType.MODEL)
        ]

    def clear(self) -> None:
        """Clear all registered components."""
        with self._lock:
            self._components.clear()

    def __len__(self) -> int:
        """Return total number of registered component versions."""
        with self._lock:
            return sum(len(versions) for versions in self._components.values())


# Global registry instance
registry = Registry()


def get_registry() -> Registry:
    """Get the global registry instance."""
    return registry
