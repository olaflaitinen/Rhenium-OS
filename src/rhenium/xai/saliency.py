"""Saliency map generation for model explanations."""

from __future__ import annotations
from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SaliencyGenerator:
    """Generate saliency maps from model gradients."""

    def __init__(self, model: nn.Module, target_layer: str | None = None):
        self.model = model
        self.target_layer = target_layer
        self._activations: dict[str, torch.Tensor] = {}
        self._gradients: dict[str, torch.Tensor] = {}
        self._hooks: list = []

    def _register_hooks(self) -> None:
        if self.target_layer is None:
            return
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                self._hooks.append(
                    module.register_forward_hook(
                        lambda m, i, o, n=name: self._activations.update({n: o})
                    )
                )
                self._hooks.append(
                    module.register_full_backward_hook(
                        lambda m, gi, go, n=name: self._gradients.update({n: go[0]})
                    )
                )

    def _remove_hooks(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
        method: Literal["gradient", "gradcam"] = "gradcam",
    ) -> np.ndarray:
        """Generate saliency map."""
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)

        if method == "gradient":
            return self._gradient_saliency(input_tensor, target_class)
        else:
            return self._gradcam(input_tensor, target_class)

    def _gradient_saliency(self, x: torch.Tensor, target: int | None) -> np.ndarray:
        output = self.model(x)
        if target is None:
            target = output.argmax(dim=1).item()
        score = output[0, target]
        score.backward()
        saliency = x.grad.abs().squeeze().cpu().numpy()
        return saliency / (saliency.max() + 1e-8)

    def _gradcam(self, x: torch.Tensor, target: int | None) -> np.ndarray:
        self._register_hooks()
        try:
            output = self.model(x)
            if target is None:
                target = output.argmax(dim=1).item()
            output[0, target].backward()

            if self.target_layer and self.target_layer in self._gradients:
                grads = self._gradients[self.target_layer]
                acts = self._activations[self.target_layer]
                weights = grads.mean(dim=(2, 3), keepdim=True)
                cam = (weights * acts).sum(dim=1, keepdim=True)
                cam = F.relu(cam)
                cam = F.interpolate(cam, x.shape[2:], mode='bilinear', align_corners=False)
                cam = cam.squeeze().cpu().numpy()
                return cam / (cam.max() + 1e-8)
            else:
                return self._gradient_saliency(x, target)
        finally:
            self._remove_hooks()
