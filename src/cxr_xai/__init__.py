"""Chest X-ray classifier spiegabile con incertezza.

Componenti:
- `data`        : caricamento PneumoniaMNIST (medmnist) e DataLoader.
- `model`       : ResNet18 con testa dropout per MC-Dropout.
- `training`    : loop di addestramento.
- `uncertainty` : predizione MC-Dropout (media + deviazione epistemica).
- `gradcam`     : mappe Grad-CAM.
- `calibration` : temperature scaling + ECE.
- `metrics`     : AUROC, F1, accuracy.
- `baseline`    : regressione logistica sui pixel.
"""

from cxr_xai.model import build_model, enable_mc_dropout

__all__ = ["build_model", "enable_mc_dropout"]
