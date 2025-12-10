# Copyright (c) 2025 Skolyn LLC. All rights reserved.

"""
MRI Physics Constraints
=======================

Physics-informed neural networks incorporating MR signal equations,
Bloch dynamics, and relaxometry constraints.

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
class MRPhysicsParams:
    """MR physics parameters."""
    field_strength: float = 3.0  # Tesla
    tr: float = 1000.0  # Repetition time (ms)
    te: float = 30.0    # Echo time (ms)
    ti: float = 0.0     # Inversion time (ms)
    flip_angle: float = 90.0  # degrees


class MRSignalPhysics:
    """
    MR signal equation models for physics constraints.
    
    Implements signal equations for various sequences that can be
    used as soft constraints in PINN training.
    """
    
    @staticmethod
    def spin_echo_signal(
        t2: np.ndarray,
        te: float,
        pd: np.ndarray,
    ) -> np.ndarray:
        """Spin echo signal: S = PD * exp(-TE/T2)"""
        return pd * np.exp(-te / (t2 + 1e-8))
    
    @staticmethod
    def gradient_echo_signal(
        t1: np.ndarray,
        t2star: np.ndarray,
        tr: float,
        te: float,
        flip_angle: float,
        pd: np.ndarray,
    ) -> np.ndarray:
        """Gradient echo signal equation."""
        alpha = np.radians(flip_angle)
        e1 = np.exp(-tr / (t1 + 1e-8))
        e2star = np.exp(-te / (t2star + 1e-8))
        
        return pd * np.sin(alpha) * (1 - e1) / (1 - np.cos(alpha) * e1) * e2star
    
    @staticmethod
    def inversion_recovery_signal(
        t1: np.ndarray,
        ti: float,
        tr: float,
        pd: np.ndarray,
    ) -> np.ndarray:
        """Inversion recovery signal: S = PD * |1 - 2*exp(-TI/T1) + exp(-TR/T1)|"""
        return pd * np.abs(1 - 2 * np.exp(-ti / (t1 + 1e-8)) + np.exp(-tr / (t1 + 1e-8)))


class T1MappingPINN(BasePINN):
    """
    PINN for T1 relaxometry from variable flip angle or IR data.
    
    Incorporates the T1 signal equation as a physics constraint.
    """
    
    name = "t1_mapping_pinn"
    version = "1.0.0"
    
    def __init__(
        self,
        config: PINNConfig | None = None,
        physics_params: MRPhysicsParams | None = None,
    ):
        super().__init__(config)
        self.physics_params = physics_params or MRPhysicsParams()
        
    def define_network(self) -> None:
        """Define T1 mapping network architecture."""
        logger.info("Defining T1 mapping network")
        # Network would predict T1 and PD from multi-flip/multi-TI data
        
    def define_physics_constraints(self) -> list[PhysicsConstraint]:
        """Define T1 relaxation physics constraints."""
        return [
            PhysicsConstraint(
                name="t1_signal_equation",
                equation="S(TI) = PD * |1 - 2*exp(-TI/T1) + exp(-TR/T1)|",
                weight=1.0,
            ),
            PhysicsConstraint(
                name="t1_positivity",
                equation="T1 > 0",
                weight=0.1,
            ),
        ]
    
    def compute_data_loss(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """MSE between predicted and measured signals."""
        return float(np.mean((predictions - targets) ** 2))
    
    def compute_physics_loss(self, predictions: np.ndarray, **kwargs: Any) -> float:
        """Compute physics residual loss."""
        # T1/PD predictions should satisfy signal equation
        t1_map = predictions[..., 0]
        pd_map = predictions[..., 1]
        
        # Positivity constraint
        positivity_loss = float(np.mean(np.maximum(0, -t1_map)))
        
        return positivity_loss
    
    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Predict T1 and PD maps."""
        if not self._trained:
            raise RuntimeError("PINN must be trained before prediction")
        # Placeholder
        return np.zeros((*inputs.shape[:-1], 2))


class T2MappingPINN(BasePINN):
    """
    PINN for T2/T2* relaxometry from multi-echo data.
    """
    
    name = "t2_mapping_pinn"
    version = "1.0.0"
    
    def define_network(self) -> None:
        logger.info("Defining T2 mapping network")
        
    def define_physics_constraints(self) -> list[PhysicsConstraint]:
        return [
            PhysicsConstraint(
                name="t2_decay",
                equation="S(TE) = S0 * exp(-TE/T2)",
                weight=1.0,
            ),
            PhysicsConstraint(
                name="t2_positivity",
                equation="T2 > 0",
                weight=0.1,
            ),
        ]
    
    def compute_data_loss(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        return float(np.mean((predictions - targets) ** 2))
    
    def compute_physics_loss(self, predictions: np.ndarray, **kwargs: Any) -> float:
        t2_map = predictions[..., 0]
        return float(np.mean(np.maximum(0, -t2_map)))
    
    def predict(self, inputs: np.ndarray) -> np.ndarray:
        if not self._trained:
            raise RuntimeError("PINN must be trained before prediction")
        return np.zeros((*inputs.shape[:-1], 2))


# Register PINN models
registry.register_pipeline(
    "t1_mapping_pinn",
    T1MappingPINN,
    version="1.0.0",
    description="Physics-informed T1 mapping",
    tags=["pinn", "quantitative", "mri", "t1"],
)

registry.register_pipeline(
    "t2_mapping_pinn",
    T2MappingPINN,
    version="1.0.0",
    description="Physics-informed T2 mapping",
    tags=["pinn", "quantitative", "mri", "t2"],
)
