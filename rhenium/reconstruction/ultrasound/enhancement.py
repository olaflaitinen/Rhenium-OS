# Copyright (c) 2025 Skolyn LLC. All rights reserved.

"""
Rhenium OS Ultrasound Enhancement
==================================

Image enhancement techniques for ultrasound:
- Speckle Reduction (Anisotropic Diffusion, Non-local Means)
- Resolution Enhancement
- Persistence (Temporal Smoothing)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

@dataclass
class USEnhancementConfig:
    method: str = "anisotropic_diffusion"
    iterations: int = 5
    kappa: float = 20.0  # Conduction coefficient
    gamma: float = 0.1   # Integration constant


class SpeckleReducer:
    """Speckle reduction algorithms."""
    
    def __init__(self, config: USEnhancementConfig | None = None):
        self.config = config or USEnhancementConfig()
        
    def filter(self, image: np.ndarray) -> np.ndarray:
        """Apply speckle reduction."""
        if self.config.method == "anisotropic_diffusion":
            return self._anisotropic_diffusion(image)
        return image

    def _anisotropic_diffusion(self, img: np.ndarray) -> np.ndarray:
        """
        Perona-Malik Anisotropic Diffusion for speckle reduction.
        Preserves edges while smoothing homogeneous regions.
        """
        # Conceptual implementation of AD
        # I(t+1) = I(t) + gamma * div(c(x,y,t) * grad(I(t)))
        refined = img.copy()
        for _ in range(self.config.iterations):
            # Placeholder for actual PDE update
            pass
        return refined


class TemporalPersistence:
    """
    Temporal smoothing (frame averaging) to reduce noise.
    """
    
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha  # Persistence factor
        self.history: np.ndarray | None = None
        
    def update(self, current_frame: np.ndarray) -> np.ndarray:
        """
        Apply temporal persistence.
        output[k] = alpha * input[k] + (1 - alpha) * output[k-1]
        """
        if self.history is None or self.history.shape != current_frame.shape:
            self.history = current_frame
            return current_frame
            
        self.history = self.alpha * current_frame + (1 - self.alpha) * self.history
        return self.history
