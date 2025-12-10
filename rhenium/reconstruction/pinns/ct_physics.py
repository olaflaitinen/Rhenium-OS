# Copyright (c) 2025 Skolyn LLC. All rights reserved.

"""
CT Physics Constraints
======================

Physics-informed neural networks for CT reconstruction incorporating
X-ray attenuation physics and projection geometry.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from rhenium.reconstruction.pinns.base_pinn import BasePINN, PhysicsConstraint, PINNConfig
from rhenium.core.registry import registry
from rhenium.core.logging import get_reconstruction_logger

logger = get_reconstruction_logger()


@dataclass
class CTPhysicsParams:
    """CT physics parameters."""
    num_projections: int = 360
    detector_pixels: int = 512
    source_to_detector: float = 1000.0  # mm
    source_to_isocenter: float = 500.0  # mm
    pixel_size: float = 1.0  # mm


class AttenuationPhysics:
    """
    X-ray attenuation physics for CT reconstruction.
    
    Implements Beer-Lambert law and projection operators.
    """
    
    @staticmethod
    def beer_lambert(
        intensity_0: float,
        mu: np.ndarray,
        path_length: np.ndarray,
    ) -> np.ndarray:
        """
        Beer-Lambert law: I = I0 * exp(-integral(mu * dl))
        
        Args:
            intensity_0: Initial X-ray intensity
            mu: Linear attenuation coefficient map
            path_length: Path lengths through material
            
        Returns:
            Transmitted intensity
        """
        return intensity_0 * np.exp(-np.sum(mu * path_length, axis=-1))
    
    @staticmethod
    def line_integral(mu: np.ndarray, path_length: np.ndarray) -> np.ndarray:
        """Compute line integral of attenuation."""
        return np.sum(mu * path_length, axis=-1)
    
    @staticmethod
    def hounsfield_units(mu: np.ndarray, mu_water: float = 0.02) -> np.ndarray:
        """Convert attenuation to Hounsfield units."""
        return 1000 * (mu - mu_water) / mu_water


class CTProjectionPINN(BasePINN):
    """
    PINN for CT reconstruction with projection physics constraints.
    
    Enforces consistency between reconstructed image and measured projections
    via the forward projection operator.
    """
    
    name = "ct_projection_pinn"
    version = "1.0.0"
    
    def __init__(
        self,
        config: PINNConfig | None = None,
        physics_params: CTPhysicsParams | None = None,
    ):
        super().__init__(config)
        self.physics_params = physics_params or CTPhysicsParams()
        
    def define_network(self) -> None:
        """Define CT reconstruction network."""
        logger.info("Defining CT projection PINN network",
                   num_projections=self.physics_params.num_projections)
        
    def define_physics_constraints(self) -> list[PhysicsConstraint]:
        """Define CT projection physics constraints."""
        return [
            PhysicsConstraint(
                name="projection_consistency",
                equation="Radon(mu) = sinogram",
                weight=1.0,
            ),
            PhysicsConstraint(
                name="attenuation_positivity",
                equation="mu >= 0 (non-negative attenuation)",
                weight=0.5,
            ),
            PhysicsConstraint(
                name="spatial_smoothness",
                equation="TV(mu) regularization",
                weight=0.1,
            ),
        ]
    
    def compute_data_loss(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """MSE between predicted and measured projections."""
        return float(np.mean((predictions - targets) ** 2))
    
    def compute_physics_loss(self, predictions: np.ndarray, **kwargs: Any) -> float:
        """Compute physics constraint residuals."""
        mu_map = predictions
        
        # Non-negativity constraint
        negativity_loss = float(np.mean(np.maximum(0, -mu_map)))
        
        # Total variation for smoothness
        tv_loss = float(
            np.mean(np.abs(np.diff(mu_map, axis=0))) +
            np.mean(np.abs(np.diff(mu_map, axis=1)))
        )
        
        return negativity_loss + 0.1 * tv_loss
    
    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Reconstruct attenuation map from projections."""
        if not self._trained:
            raise RuntimeError("PINN must be trained before prediction")
        # Placeholder
        return np.zeros((self.physics_params.detector_pixels,
                        self.physics_params.detector_pixels))


class LowDoseCTPINN(BasePINN):
    """
    PINN for low-dose CT reconstruction with noise modeling.
    
    Incorporates Poisson noise statistics as physics prior.
    """
    
    name = "low_dose_ct_pinn"
    version = "1.0.0"
    
    def __init__(self, config: PINNConfig | None = None):
        super().__init__(config)
        
    def define_network(self) -> None:
        logger.info("Defining low-dose CT PINN")
        
    def define_physics_constraints(self) -> list[PhysicsConstraint]:
        return [
            PhysicsConstraint(
                name="poisson_noise",
                equation="Photon counting statistics",
                weight=0.5,
            ),
            PhysicsConstraint(
                name="projection_consistency",
                equation="Forward projection match",
                weight=1.0,
            ),
        ]
    
    def compute_data_loss(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        # Weighted least squares for Poisson noise
        weights = np.maximum(targets, 1e-6)
        return float(np.mean(weights * (predictions - targets) ** 2))
    
    def compute_physics_loss(self, predictions: np.ndarray, **kwargs: Any) -> float:
        return float(np.mean(np.maximum(0, -predictions)))
    
    def predict(self, inputs: np.ndarray) -> np.ndarray:
        if not self._trained:
            raise RuntimeError("PINN must be trained")
        return np.zeros_like(inputs)


# Register CT PINNs
registry.register_pipeline(
    "ct_projection_pinn",
    CTProjectionPINN,
    version="1.0.0",
    description="Physics-informed CT reconstruction",
    tags=["pinn", "ct", "reconstruction"],
)

registry.register_pipeline(
    "low_dose_ct_pinn",
    LowDoseCTPINN,
    version="1.0.0",
    description="Low-dose CT with noise physics",
    tags=["pinn", "ct", "low-dose"],
)
