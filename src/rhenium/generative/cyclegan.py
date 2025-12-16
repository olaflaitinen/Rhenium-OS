"""CycleGAN unpaired image-to-image translation."""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from rhenium.core.registry import registry


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch, ch, 3),
            nn.InstanceNorm2d(ch),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch, ch, 3),
            nn.InstanceNorm2d(ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


@registry.register("generator", "cyclegan", version="1.0.0")
class CycleGANGenerator(nn.Module):
    """ResNet-based generator for CycleGAN."""

    def __init__(self, in_ch: int = 1, out_ch: int = 1, features: int = 64, n_res: int = 9):
        super().__init__()
        # Encoder
        self.enc = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, features, 7),
            nn.InstanceNorm2d(features),
            nn.ReLU(),
            nn.Conv2d(features, features * 2, 3, stride=2, padding=1),
            nn.InstanceNorm2d(features * 2),
            nn.ReLU(),
            nn.Conv2d(features * 2, features * 4, 3, stride=2, padding=1),
            nn.InstanceNorm2d(features * 4),
            nn.ReLU(),
        )
        # Residual blocks
        self.res = nn.Sequential(*[ResBlock(features * 4) for _ in range(n_res)])
        # Decoder
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(features * 4, features * 2, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(features * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(features * 2, features, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(features),
            nn.ReLU(),
            nn.ReflectionPad2d(3),
            nn.Conv2d(features, out_ch, 7),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dec(self.res(self.enc(x)))


class CycleGAN:
    """CycleGAN training wrapper."""

    def __init__(
        self,
        G_AB: nn.Module,
        G_BA: nn.Module,
        D_A: nn.Module,
        D_B: nn.Module,
        lambda_cyc: float = 10.0,
        lambda_id: float = 0.5,
    ):
        self.G_AB = G_AB
        self.G_BA = G_BA
        self.D_A = D_A
        self.D_B = D_B
        self.lambda_cyc = lambda_cyc
        self.lambda_id = lambda_id

    def generator_loss(self, real_A: torch.Tensor, real_B: torch.Tensor) -> dict[str, torch.Tensor]:
        fake_B = self.G_AB(real_A)
        fake_A = self.G_BA(real_B)
        rec_A = self.G_BA(fake_B)
        rec_B = self.G_AB(fake_A)

        # GAN loss
        loss_G_AB = F.mse_loss(self.D_B(fake_B), torch.ones_like(self.D_B(fake_B)))
        loss_G_BA = F.mse_loss(self.D_A(fake_A), torch.ones_like(self.D_A(fake_A)))

        # Cycle loss
        loss_cyc = F.l1_loss(rec_A, real_A) + F.l1_loss(rec_B, real_B)

        total = loss_G_AB + loss_G_BA + self.lambda_cyc * loss_cyc
        return {"total": total, "gan": loss_G_AB + loss_G_BA, "cycle": loss_cyc}
