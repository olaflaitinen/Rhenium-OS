"""MedGemma VLM client implementations."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class MedGemmaClient(ABC):
    """Abstract client for MedGemma VLM."""

    @abstractmethod
    def generate_narrative(
        self,
        image: np.ndarray,
        findings: dict[str, Any],
        prompt: str = "",
    ) -> str:
        """Generate narrative explanation."""
        pass


class StubClient(MedGemmaClient):
    """Stub client for testing without MedGemma."""

    def generate_narrative(
        self,
        image: np.ndarray,
        findings: dict[str, Any],
        prompt: str = "",
    ) -> str:
        return (
            f"AI analysis detected {len(findings.get('detections', []))} finding(s). "
            "This is a stub response for testing. "
            "Clinical correlation is recommended."
        )


class LocalClient(MedGemmaClient):
    """Local MedGemma inference client."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self._model = None

    def generate_narrative(
        self,
        image: np.ndarray,
        findings: dict[str, Any],
        prompt: str = "",
    ) -> str:
        # Placeholder for local inference
        return "LocalClient narrative generation not yet implemented."


class RemoteClient(MedGemmaClient):
    """Remote MedGemma API client."""

    def __init__(self, endpoint: str, api_key: str = ""):
        self.endpoint = endpoint
        self.api_key = api_key

    def generate_narrative(
        self,
        image: np.ndarray,
        findings: dict[str, Any],
        prompt: str = "",
    ) -> str:
        # Placeholder for remote API call
        return "RemoteClient narrative generation not yet implemented."


def get_client(backend: str = "stub", **kwargs: Any) -> MedGemmaClient:
    """Factory for MedGemma clients."""
    if backend == "stub":
        return StubClient()
    elif backend == "local":
        return LocalClient(kwargs.get("model_path", ""))
    elif backend == "remote":
        return RemoteClient(kwargs.get("endpoint", ""), kwargs.get("api_key", ""))
    else:
        return StubClient()
