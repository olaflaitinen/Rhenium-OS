# Copyright (c) 2025 Skolyn LLC. All rights reserved.

"""
Physics-Informed Neural Networks (PINNs) Module
================================================

Abstractions for physics-constrained deep learning in medical imaging
reconstruction and quantitative mapping.

"""

from rhenium.reconstruction.pinns.base_pinn import (
    BasePINN,
    PhysicsConstraint,
    PINNConfig,
)
from rhenium.reconstruction.pinns.mri_physics import (
    MRSignalPhysics,
    T1MappingPINN,
    T2MappingPINN,
)
from rhenium.reconstruction.pinns.ct_physics import (
    AttenuationPhysics,
    CTProjectionPINN,
)

__all__ = [
    "BasePINN",
    "PhysicsConstraint",
    "PINNConfig",
    "MRSignalPhysics",
    "T1MappingPINN",
    "T2MappingPINN",
    "AttenuationPhysics",
    "CTProjectionPINN",
]
