"""
Rhenium OS Physics-Informed Neural Networks (PINNs).

Provides physics-based losses and constraints for reconstruction.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Callable, Literal
import torch
import torch.nn as nn
import numpy as np
from rhenium.reconstruction.base import fft2c, ifft2c


class BasePINN(nn.Module, ABC):
    """Base class for Physics-Informed Neural Networks."""

    def __init__(self, backbone: nn.Module, physics_loss: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.physics_loss = physics_loss

    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        pass

    def compute_total_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        physics_params: dict[str, Any],
        lambda_data: float = 1.0,
        lambda_physics: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        data_loss = nn.functional.mse_loss(pred, target)
        physics_loss = self.physics_loss(pred, **physics_params)
        total = lambda_data * data_loss + lambda_physics * physics_loss
        return {"total": total, "data": data_loss, "physics": physics_loss}


class MRIPINNLoss(nn.Module):
    """Physics loss for MRI reconstruction."""

    def __init__(self, lambda_dc: float = 1.0, lambda_smooth: float = 0.01):
        super().__init__()
        self.lambda_dc = lambda_dc
        self.lambda_smooth = lambda_smooth

    def forward(
        self,
        pred: torch.Tensor,
        kspace: torch.Tensor,
        mask: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        # Data consistency in k-space
        pred_kspace = fft2c(pred.to(torch.complex64))
        dc_error = mask * (pred_kspace - kspace)
        loss_dc = torch.norm(dc_error, p=2) ** 2 / pred.numel()

        # Smoothness (TV)
        dx = torch.diff(pred, dim=-1)
        dy = torch.diff(pred, dim=-2)
        loss_tv = (dx.abs().mean() + dy.abs().mean())

        return self.lambda_dc * loss_dc + self.lambda_smooth * loss_tv


class CTPINNLoss(nn.Module):
    """Physics loss for CT reconstruction."""

    def __init__(self, lambda_proj: float = 1.0, lambda_tv: float = 0.01):
        super().__init__()
        self.lambda_proj = lambda_proj
        self.lambda_tv = lambda_tv

    def forward(
        self,
        pred: torch.Tensor,
        sinogram: torch.Tensor,
        angles: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        # Projection consistency (simplified)
        pred_sino = self._forward_project(pred, angles)
        loss_proj = nn.functional.mse_loss(pred_sino, sinogram)

        # TV regularization
        dx = torch.diff(pred, dim=-1)
        dy = torch.diff(pred, dim=-2)
        loss_tv = torch.sqrt(dx**2 + dy**2 + 1e-8).mean()

        return self.lambda_proj * loss_proj + self.lambda_tv * loss_tv

    def _forward_project(self, image: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        B = image.shape[0]
        sino = []
        for theta in angles:
            rot = self._rotate(image, theta)
            sino.append(rot.sum(dim=-2))
        return torch.stack(sino, dim=1)

    def _rotate(self, x: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        cos_a, sin_a = torch.cos(angle), torch.sin(angle)
        theta = torch.tensor([[cos_a, -sin_a, 0], [sin_a, cos_a, 0]], device=x.device)
        grid = nn.functional.affine_grid(theta.unsqueeze(0), x.unsqueeze(1).shape, align_corners=False)
        return nn.functional.grid_sample(x.unsqueeze(1), grid, align_corners=False).squeeze(1)


class CollocationSampler:
    """Samples collocation points for PINN training."""

    @staticmethod
    def uniform_grid(shape: tuple[int, ...], device: str = "cpu") -> torch.Tensor:
        grids = [torch.linspace(0, 1, s, device=device) for s in shape]
        return torch.stack(torch.meshgrid(*grids, indexing='ij'), dim=-1).reshape(-1, len(shape))

    @staticmethod
    def random_uniform(n: int, dims: int, device: str = "cpu") -> torch.Tensor:
        return torch.rand(n, dims, device=device)

    @staticmethod
    def adaptive(residuals: torch.Tensor, n: int, alpha: float = 2.0) -> torch.Tensor:
        probs = residuals.abs() ** alpha
        probs = probs / probs.sum()
        indices = torch.multinomial(probs.flatten(), n, replacement=True)
        return indices


class AdaptiveLossWeighting(nn.Module):
    """Automatically balance multiple loss terms."""

    def __init__(self, num_losses: int, alpha: float = 0.1):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_losses))
        self.alpha = alpha

    def forward(self, losses: list[torch.Tensor]) -> torch.Tensor:
        return sum(w * l for w, l in zip(self.weights, losses))
