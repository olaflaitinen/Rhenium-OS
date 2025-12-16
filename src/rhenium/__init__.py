"""
Rhenium OS - Multi-Modality AI Platform for Medical Imaging Research.

This package provides tools for medical imaging AI research including:
- Reconstruction (MRI, CT, Ultrasound, X-ray)
- Perception (segmentation, detection, classification)
- Generative models (GANs, super-resolution)
- Explainability (Evidence Dossier)
- Governance (model cards, audit logging)

Note: This software is for research purposes only and is not approved
for clinical use.
"""

__version__ = "0.1.0"
__author__ = "Rhenium OS Team"
__license__ = "EUPL-1.1"

# Public API exports
from rhenium.core.config import get_settings, RheniumSettings
from rhenium.core.registry import registry, get_registry

__all__ = [
    "__version__",
    "__author__",
    "__license__",
    "get_settings",
    "RheniumSettings",
    "registry",
    "get_registry",
]
