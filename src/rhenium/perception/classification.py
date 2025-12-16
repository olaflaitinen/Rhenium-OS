"""
Rhenium OS 3D Classification Models.

This module provides classification architectures for image-level
and region-level predictions including binary, multi-class, and ordinal.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from rhenium.core.registry import registry


class ResidualBlock3D(nn.Module):
    """3D Residual block with optional downsampling."""

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ):
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3,
            padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


@registry.register("model", "resnet3d", version="1.0.0", description="3D ResNet for classification")
class ResNet3D(nn.Module):
    """
    3D ResNet for volumetric classification.

    Args:
        in_channels: Number of input channels
        num_classes: Number of output classes
        base_features: Number of features in first layer
        layers: Number of residual blocks per stage
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        base_features: int = 64,
        layers: list[int] | None = None,
    ):
        super().__init__()

        if layers is None:
            layers = [2, 2, 2, 2]  # ResNet-18 style

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.inplanes = base_features

        # Stem
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, base_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(base_features),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )

        # Residual layers
        self.layer1 = self._make_layer(base_features, layers[0])
        self.layer2 = self._make_layer(base_features * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(base_features * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(base_features * 8, layers[3], stride=2)

        # Classification head
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(base_features * 8, num_classes)

        self._init_weights()

    def _make_layer(
        self,
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        """Create a residual layer."""
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes),
            )

        layers = [ResidualBlock3D(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes

        for _ in range(1, blocks):
            layers.append(ResidualBlock3D(planes, planes))

        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, D, H, W)

        Returns:
            Logits tensor of shape (B, num_classes)
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)

        return x

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get class probabilities."""
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get predicted class indices."""
        with torch.no_grad():
            logits = self.forward(x)
            return logits.argmax(dim=1)


@registry.register("model", "ordinal_classifier", version="1.0.0", description="Ordinal classification for grading")
class OrdinalClassifier(nn.Module):
    """
    Ordinal classifier for grading scales (PI-RADS, BI-RADS, etc.).

    Uses threshold-based ordinal regression for ordered classes.

    Args:
        backbone: Feature extraction backbone
        num_grades: Number of ordinal grades
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_grades: int,
        backbone_out_features: int = 512,
    ):
        super().__init__()

        self.backbone = backbone
        self.num_grades = num_grades
        self.num_thresholds = num_grades - 1

        # Learnable thresholds
        self.thresholds = nn.Parameter(torch.zeros(self.num_thresholds))

        # Projection from backbone features
        self.fc = nn.Linear(backbone_out_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor

        Returns:
            Grade probabilities of shape (B, num_grades)
        """
        # Get backbone features
        features = self.backbone(x)

        # Global pooling if needed
        if features.dim() > 2:
            features = F.adaptive_avg_pool3d(features, 1).flatten(1)

        # Projection to scalar
        logits = self.fc(features).squeeze(-1)  # (B,)

        # Compute cumulative probabilities
        # P(Y > k) = sigmoid(logits - threshold_k)
        cumprobs = torch.sigmoid(
            logits.unsqueeze(-1) - self.thresholds.unsqueeze(0)
        )  # (B, num_thresholds)

        # Convert to grade probabilities
        # P(Y = 0) = 1 - P(Y > 0)
        # P(Y = k) = P(Y > k-1) - P(Y > k) for 0 < k < num_grades-1
        # P(Y = num_grades-1) = P(Y > num_grades-2)
        probs = torch.zeros(x.shape[0], self.num_grades, device=x.device)
        probs[:, 0] = 1 - cumprobs[:, 0]
        for k in range(1, self.num_grades - 1):
            probs[:, k] = cumprobs[:, k - 1] - cumprobs[:, k]
        probs[:, -1] = cumprobs[:, -1]

        return probs

    def predict_grade(self, x: torch.Tensor) -> torch.Tensor:
        """Get predicted grade."""
        with torch.no_grad():
            probs = self.forward(x)
            return probs.argmax(dim=1)


@registry.register("model", "densenet3d", version="1.0.0", description="3D DenseNet for classification")
class DenseNet3D(nn.Module):
    """
    Simplified 3D DenseNet for classification.

    Args:
        in_channels: Number of input channels
        num_classes: Number of output classes
        growth_rate: Feature growth rate per layer
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        growth_rate: int = 32,
        num_layers: list[int] | None = None,
    ):
        super().__init__()

        if num_layers is None:
            num_layers = [6, 12, 24, 16]

        init_features = growth_rate * 2

        # Initial convolution
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )

        num_features = init_features

        # Dense blocks (simplified - using conv blocks)
        for i, num in enumerate(num_layers):
            block = self._make_dense_block(num_features, growth_rate, num)
            self.features.add_module(f'denseblock{i+1}', block)
            num_features = num_features + num * growth_rate

            if i < len(num_layers) - 1:
                trans = self._make_transition(num_features, num_features // 2)
                self.features.add_module(f'transition{i+1}', trans)
                num_features = num_features // 2

        self.features.add_module('norm_final', nn.BatchNorm3d(num_features))

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Linear(num_features, num_classes)

    def _make_dense_block(
        self,
        in_features: int,
        growth_rate: int,
        num_layers: int,
    ) -> nn.Sequential:
        """Create a dense block (simplified)."""
        layers = []
        for i in range(num_layers):
            layers.append(nn.Sequential(
                nn.BatchNorm3d(in_features + i * growth_rate),
                nn.ReLU(inplace=True),
                nn.Conv3d(in_features + i * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False),
            ))
        return nn.Sequential(*layers)

    def _make_transition(self, in_features: int, out_features: int) -> nn.Sequential:
        """Create a transition layer."""
        return nn.Sequential(
            nn.BatchNorm3d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_features, out_features, kernel_size=1, bias=False),
            nn.AvgPool3d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = self.avgpool(out)
        out = out.flatten(1)
        out = self.classifier(out)
        return out


def load_classification_model(
    name: str,
    checkpoint: Path | str | None = None,
    device: str = "cuda",
    **kwargs: Any,
) -> nn.Module:
    """
    Load a classification model from registry.

    Args:
        name: Model name
        checkpoint: Path to checkpoint
        device: Target device
        **kwargs: Model configuration

    Returns:
        Loaded model
    """
    model_cls = registry.get("model", name)
    model = model_cls(**kwargs)

    if checkpoint is not None:
        checkpoint = Path(checkpoint)
        if checkpoint.exists():
            state_dict = torch.load(checkpoint, map_location=device)
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            model.load_state_dict(state_dict)

    model = model.to(device)
    return model
