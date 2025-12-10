# Copyright (c) 2025 Skolyn LLC. All rights reserved.

"""
Skolyn Rhenium OS
=================

State-of-the-Art AI Operating System for Diagnostic Medical Imaging.

Rhenium OS is Skolyn's centralized, production-grade artificial intelligence
operating system engineered for diagnostic radiology. Built on a foundation
of proprietary deep learning models, including Physics-Informed Neural Networks
(PINNs), Generative Adversarial Networks (GANs), U-Net architectures, Vision
Transformers, and 3D CNNs. Rhenium OS delivers unprecedented speed, accuracy,
and transparency across MRI, CT, Ultrasound, and X-ray modalities.

"""

from importlib.metadata import version, PackageNotFoundError


try:
    __version__ = version("rhenium-os")
except PackageNotFoundError:
    __version__ = "0.1.0.dev0"

__author__ = "Skolyn LLC"
__email__ = "engineering@skolyn.com"
__license__ = "EUPL-1.1"

# Core module exports
from rhenium.core.config import get_settings, RheniumSettings
from rhenium.core.errors import RheniumError
from rhenium.core.logging import get_logger
from rhenium.core.registry import registry


__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "get_settings",
    "RheniumSettings",
    "RheniumError",
    "get_logger",
    "registry",
]
