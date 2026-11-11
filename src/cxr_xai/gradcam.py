"""Grad-CAM to explain predictions (lightweight implementation, no extra dependencies).

Grad-CAM weights the activation maps of a convolutional layer by the spatial
mean of the target-class gradients, then applies a ReLU. The resulting heatmap
highlights the regions that drove the prediction.

Reference: Selvaraju R.R. et al., "Grad-CAM: Visual Explanations from Deep
Networks via Gradient-based Localization", ICCV 2017. DOI: 10.1109/ICCV.2017.74
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """Compute Grad-CAM on a target layer (default: last conv block of a ResNet)."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, _module, _inp, out):
        self.activations = out.detach()

    def _save_gradients(self, _module, _grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, x: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        """Heatmap (H, W) normalised to [0,1] for a single input (1×C×H×W)."""
        self.model.eval()
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        self.model.zero_grad()
        logits[0, class_idx].backward()

        grads = self.gradients  # (1, K, h, w)
        acts = self.activations  # (1, K, h, w)
        weights = grads.mean(dim=(2, 3), keepdim=True)  # (1, K, 1, 1)
        cam = F.relu((weights * acts).sum(dim=1, keepdim=True))  # (1,1,h,w)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        cam -= cam.min()
        denom = cam.max()
        if denom > 0:
            cam /= denom
        return cam


def resnet_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    """Return the last convolutional block of a ResNet (layer4)."""
    return model.layer4[-1]
