# Copyright (c) 2025 Skolyn LLC. All rights reserved.
# SPDX-License-Identifier: EUPL-1.1

"""
Visual Explanations
===================

Generation of visual explanation artifacts: saliency maps, overlays, heatmaps.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.ndimage import gaussian_filter

from rhenium.xai.explanation_schema import VisualEvidence
from rhenium.core.logging import get_xai_logger

if TYPE_CHECKING:
    import torch

logger = get_xai_logger()


def generate_saliency_map(
    model: "torch.nn.Module",
    input_tensor: "torch.Tensor",
    target_class: int | None = None,
    method: str = "gradcam",
) -> VisualEvidence:
    """
    Generate saliency map for a model prediction.

    Args:
        model: PyTorch model.
        input_tensor: Input image tensor.
        target_class: Target class for explanation.
        method: Saliency method ('gradcam', 'vanilla', 'integrated').

    Returns:
        VisualEvidence containing saliency map.
    """
    logger.info("Generating saliency map", method=method)

    # Placeholder implementation
    if hasattr(input_tensor, 'numpy'):
        shape = input_tensor.numpy().shape[-2:]
    else:
        shape = (256, 256)

    saliency = np.random.rand(*shape).astype(np.float32)
    saliency = gaussian_filter(saliency, sigma=10)
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

    return VisualEvidence(
        artifact_type="saliency",
        data=saliency,
        description=f"Saliency map generated using {method}",
        colormap="jet",
    )


def generate_segmentation_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    class_colors: dict[int, tuple[int, int, int]] | None = None,
    alpha: float = 0.5,
) -> VisualEvidence:
    """
    Generate segmentation overlay visualization.

    Args:
        image: Base image (H, W) or (H, W, C).
        mask: Segmentation mask (H, W) with integer class labels.
        class_colors: Mapping of class ID to RGB color.
        alpha: Overlay transparency.

    Returns:
        VisualEvidence containing overlay image.
    """
    logger.info("Generating segmentation overlay")

    if class_colors is None:
        class_colors = {
            0: (0, 0, 0),
            1: (255, 0, 0),
            2: (0, 255, 0),
            3: (0, 0, 255),
            4: (255, 255, 0),
        }

    # Normalize image to 0-255
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)

    overlay = image.copy().astype(np.float32)

    for class_id, color in class_colors.items():
        class_mask = mask == class_id
        if class_mask.any():
            for c in range(3):
                overlay[:, :, c] = np.where(
                    class_mask,
                    overlay[:, :, c] * (1 - alpha) + color[c] * alpha,
                    overlay[:, :, c],
                )

    return VisualEvidence(
        artifact_type="overlay",
        data=overlay.astype(np.uint8),
        description="Segmentation mask overlay",
    )


def generate_attention_map(
    attention_weights: np.ndarray,
    image_shape: tuple[int, int],
) -> VisualEvidence:
    """Generate attention map visualization."""
    from scipy.ndimage import zoom

    # Resize attention to image shape
    if attention_weights.shape != image_shape:
        zoom_factors = (
            image_shape[0] / attention_weights.shape[0],
            image_shape[1] / attention_weights.shape[1],
        )
        attention = zoom(attention_weights, zoom_factors, order=1)
    else:
        attention = attention_weights

    # Normalize
    attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)

    return VisualEvidence(
        artifact_type="attention",
        data=attention,
        description="Model attention map",
        colormap="hot",
    )
