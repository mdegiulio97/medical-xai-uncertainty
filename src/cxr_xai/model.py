"""Model: ResNet18 with a dropout head for MC-Dropout.

The network reuses a ResNet18 backbone (ImageNet weights when available) and
replaces the final classifier with `Dropout -> Linear`. Dropout is then kept
active at inference to estimate epistemic uncertainty (MC-Dropout).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


def build_model(num_classes: int = 2, dropout: float = 0.3, pretrained: bool = True) -> nn.Module:
    """Build a ResNet18 with a dropout head.

    Parameters
    ----------
    num_classes : number of classes (PneumoniaMNIST: 2).
    dropout : dropout probability in the head (also used at inference for MC).
    pretrained : if True, load the ImageNet weights (needs network access on first use).
    """
    weights = None
    if pretrained:
        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
        except Exception:  # pragma: no cover - offline / torchvision version
            weights = None
    net = models.resnet18(weights=weights)
    in_features = net.fc.in_features
    net.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return net


def enable_mc_dropout(model: nn.Module) -> None:
    """Put the model in eval mode but re-enable the Dropout layers (for MC-Dropout)."""
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
