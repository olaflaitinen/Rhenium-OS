# Copyright (c) 2025 Skolyn LLC. All rights reserved.

"""
Rhenium OS Component Registry
=============================

A registry pattern implementation for managing pipelines, models, modalities,
and organ modules. Supports version-aware registration and lookup, enabling
runtime composition and backwards compatibility.

Usage:
    from rhenium.core.registry import registry, register_pipeline

    # Register using decorator
    @register_pipeline("mri_knee_v1", version="1.0.0")
    class KneeMRIPipeline(BasePipeline):
        pass

    # Look up component
    pipeline_cls = registry.get_pipeline("mri_knee_v1", version="latest")
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, TypeVar

from rhenium.core.errors import RegistryError


class ComponentType(str, Enum):
    """Types of components that can be registered."""

    PIPELINE = "pipeline"
    MODEL = "model"
    MODALITY = "modality"
    ORGAN_MODULE = "organ_module"
    PREPROCESSOR = "preprocessor"
    RECONSTRUCTION = "reconstruction"
    XAI_GENERATOR = "xai_generator"
    PROMPT_TEMPLATE = "prompt_template"


@dataclass
class ComponentMetadata:
    """Metadata for a registered component."""

    name: str
    component_type: ComponentType
    version: str
    component_class: type | Callable[..., Any]
    description: str = ""
    author: str = ""
    registered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tags: list[str] = field(default_factory=list)
    deprecated: bool = False
    deprecation_message: str = ""

    def __post_init__(self) -> None:
        """Validate version format."""
        if not self._is_valid_version(self.version):
            raise RegistryError(
                f"Invalid version format: {self.version}. Use semantic versioning (e.g., 1.0.0).",
                component_type=self.component_type.value,
                component_name=self.name,
            )

    @staticmethod
    def _is_valid_version(version: str) -> bool:
        """Check if version follows semantic versioning."""
        pattern = r"^\d+\.\d+\.\d+(-[a-zA-Z0-9]+(\.[a-zA-Z0-9]+)*)?$"
        return bool(re.match(pattern, version))


T = TypeVar("T")


class ComponentRegistry:
    """
    Central registry for Rhenium OS components.

    The registry maintains a versioned catalog of all registered components,
    enabling runtime lookup and composition of pipelines, models, and modules.
    """

    def __init__(self) -> None:
        """Initialize the component registry."""
        # Structure: {component_type: {name: {version: metadata}}}
        self._registry: dict[ComponentType, dict[str, dict[str, ComponentMetadata]]] = defaultdict(
            lambda: defaultdict(dict)
        )

    def register(
        self,
        name: str,
        component_type: ComponentType,
        component_class: type | Callable[..., Any],
        version: str = "1.0.0",
        description: str = "",
        author: str = "",
        tags: list[str] | None = None,
    ) -> None:
        """
        Register a component with the registry.

        Args:
            name: Unique name for the component within its type.
            component_type: Type of component being registered.
            component_class: The class or factory function for the component.
            version: Semantic version string (e.g., "1.0.0").
            description: Human-readable description of the component.
            author: Author or team responsible for the component.
            tags: Optional list of tags for categorization.

        Raises:
            RegistryError: If the component version is already registered.
        """
        if version in self._registry[component_type][name]:
            raise RegistryError(
                f"Component '{name}' version '{version}' is already registered.",
                component_type=component_type.value,
                component_name=name,
            )

        metadata = ComponentMetadata(
            name=name,
            component_type=component_type,
            version=version,
            component_class=component_class,
            description=description,
            author=author,
            tags=tags or [],
        )

        self._registry[component_type][name][version] = metadata

    def get(
        self,
        name: str,
        component_type: ComponentType,
        version: str = "latest",
    ) -> type | Callable[..., Any]:
        """
        Retrieve a component from the registry.

        Args:
            name: Name of the component.
            component_type: Type of component to retrieve.
            version: Version to retrieve ("latest" for most recent).

        Returns:
            The component class or factory function.

        Raises:
            RegistryError: If the component is not found.
        """
        if name not in self._registry[component_type]:
            raise RegistryError(
                f"Component '{name}' of type '{component_type.value}' not found.",
                component_type=component_type.value,
                component_name=name,
            )

        versions = self._registry[component_type][name]

        if version == "latest":
            # Get the highest version
            sorted_versions = sorted(versions.keys(), key=self._parse_version, reverse=True)
            version = sorted_versions[0]
        elif version not in versions:
            raise RegistryError(
                f"Version '{version}' of component '{name}' not found.",
                component_type=component_type.value,
                component_name=name,
            )

        return versions[version].component_class

    def get_metadata(
        self,
        name: str,
        component_type: ComponentType,
        version: str = "latest",
    ) -> ComponentMetadata:
        """
        Retrieve component metadata.

        Args:
            name: Name of the component.
            component_type: Type of component.
            version: Version to retrieve ("latest" for most recent).

        Returns:
            ComponentMetadata for the specified component.

        Raises:
            RegistryError: If the component is not found.
        """
        if name not in self._registry[component_type]:
            raise RegistryError(
                f"Component '{name}' of type '{component_type.value}' not found.",
                component_type=component_type.value,
                component_name=name,
            )

        versions = self._registry[component_type][name]

        if version == "latest":
            sorted_versions = sorted(versions.keys(), key=self._parse_version, reverse=True)
            version = sorted_versions[0]
        elif version not in versions:
            raise RegistryError(
                f"Version '{version}' of component '{name}' not found.",
                component_type=component_type.value,
                component_name=name,
            )

        return versions[version]

    def list_components(
        self,
        component_type: ComponentType | None = None,
        tags: list[str] | None = None,
    ) -> list[ComponentMetadata]:
        """
        List registered components.

        Args:
            component_type: Filter by component type (None for all).
            tags: Filter by tags (components must have all specified tags).

        Returns:
            List of ComponentMetadata for matching components.
        """
        result: list[ComponentMetadata] = []

        types_to_check = [component_type] if component_type else list(ComponentType)

        for ctype in types_to_check:
            for name, versions in self._registry[ctype].items():
                for version, metadata in versions.items():
                    if tags:
                        if not all(tag in metadata.tags for tag in tags):
                            continue
                    result.append(metadata)

        return result

    def deprecate(
        self,
        name: str,
        component_type: ComponentType,
        version: str,
        message: str = "",
    ) -> None:
        """
        Mark a component version as deprecated.

        Args:
            name: Name of the component.
            component_type: Type of component.
            version: Version to deprecate.
            message: Deprecation message with migration guidance.

        Raises:
            RegistryError: If the component is not found.
        """
        metadata = self.get_metadata(name, component_type, version)
        metadata.deprecated = True
        metadata.deprecation_message = message

    def unregister(
        self,
        name: str,
        component_type: ComponentType,
        version: str | None = None,
    ) -> None:
        """
        Remove a component from the registry.

        Args:
            name: Name of the component.
            component_type: Type of component.
            version: Version to remove (None to remove all versions).

        Raises:
            RegistryError: If the component is not found.
        """
        if name not in self._registry[component_type]:
            raise RegistryError(
                f"Component '{name}' of type '{component_type.value}' not found.",
                component_type=component_type.value,
                component_name=name,
            )

        if version:
            if version not in self._registry[component_type][name]:
                raise RegistryError(
                    f"Version '{version}' of component '{name}' not found.",
                    component_type=component_type.value,
                    component_name=name,
                )
            del self._registry[component_type][name][version]
        else:
            del self._registry[component_type][name]

    @staticmethod
    def _parse_version(version: str) -> tuple[int, int, int, str]:
        """Parse version string for comparison."""
        match = re.match(r"^(\d+)\.(\d+)\.(\d+)(-.*)?$", version)
        if match:
            major, minor, patch = int(match.group(1)), int(match.group(2)), int(match.group(3))
            prerelease = match.group(4) or ""
            return (major, minor, patch, prerelease)
        return (0, 0, 0, version)

    # Convenience methods for specific component types

    def register_pipeline(
        self,
        name: str,
        pipeline_class: type,
        version: str = "1.0.0",
        **kwargs: Any,
    ) -> None:
        """Register a pipeline component."""
        self.register(name, ComponentType.PIPELINE, pipeline_class, version, **kwargs)

    def get_pipeline(self, name: str, version: str = "latest") -> type:
        """Get a pipeline component."""
        return self.get(name, ComponentType.PIPELINE, version)

    def register_model(
        self,
        name: str,
        model_class: type,
        version: str = "1.0.0",
        **kwargs: Any,
    ) -> None:
        """Register a model component."""
        self.register(name, ComponentType.MODEL, model_class, version, **kwargs)

    def get_model(self, name: str, version: str = "latest") -> type:
        """Get a model component."""
        return self.get(name, ComponentType.MODEL, version)

    def register_organ_module(
        self,
        name: str,
        module_class: type,
        version: str = "1.0.0",
        **kwargs: Any,
    ) -> None:
        """Register an organ module component."""
        self.register(name, ComponentType.ORGAN_MODULE, module_class, version, **kwargs)

    def get_organ_module(self, name: str, version: str = "latest") -> type:
        """Get an organ module component."""
        return self.get(name, ComponentType.ORGAN_MODULE, version)


# Global registry instance
registry = ComponentRegistry()


# Decorator factories for convenient registration
def register_pipeline(
    name: str,
    version: str = "1.0.0",
    description: str = "",
    author: str = "",
    tags: list[str] | None = None,
) -> Callable[[type], type]:
    """Decorator to register a pipeline class."""

    def decorator(cls: type) -> type:
        registry.register_pipeline(
            name=name,
            pipeline_class=cls,
            version=version,
            description=description,
            author=author,
            tags=tags,
        )
        return cls

    return decorator


def register_model(
    name: str,
    version: str = "1.0.0",
    description: str = "",
    author: str = "",
    tags: list[str] | None = None,
) -> Callable[[type], type]:
    """Decorator to register a model class."""

    def decorator(cls: type) -> type:
        registry.register_model(
            name=name,
            model_class=cls,
            version=version,
            description=description,
            author=author,
            tags=tags,
        )
        return cls

    return decorator


def register_organ_module(
    name: str,
    version: str = "1.0.0",
    description: str = "",
    author: str = "",
    tags: list[str] | None = None,
) -> Callable[[type], type]:
    """Decorator to register an organ module class."""

    def decorator(cls: type) -> type:
        registry.register_organ_module(
            name=name,
            module_class=cls,
            version=version,
            description=description,
            author=author,
            tags=tags,
        )
        return cls

    return decorator
