"""Explainable chest X-ray classifier with uncertainty.

Components:
- `data`        : PneumoniaMNIST loading (medmnist) and DataLoaders.
- `model`       : ResNet18 with a dropout head for MC-Dropout.
- `training`    : training loop.
- `uncertainty` : MC-Dropout prediction (mean + epistemic standard deviation).
- `gradcam`     : Grad-CAM maps.
- `calibration` : temperature scaling + ECE.
- `metrics`     : AUROC, F1, accuracy.
- `baseline`    : logistic regression on pixels.
"""

from cxr_xai.model import build_model, enable_mc_dropout

__all__ = ["build_model", "enable_mc_dropout"]
