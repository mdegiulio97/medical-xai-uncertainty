"""Modello: ResNet18 con testa dropout per MC-Dropout.

La rete riusa un backbone ResNet18 (pesi ImageNet se disponibili) e sostituisce
il classificatore finale con `Dropout -> Linear`. Il dropout viene poi tenuto
attivo a inferenza per stimare l'incertezza epistemica (MC-Dropout).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


def build_model(num_classes: int = 2, dropout: float = 0.3, pretrained: bool = True) -> nn.Module:
    """Costruisce ResNet18 con testa dropout.

    Parametri
    ---------
    num_classes : numero di classi (PneumoniaMNIST: 2).
    dropout : probabilità di dropout nella testa (usata anche a inferenza per MC).
    pretrained : se True carica i pesi ImageNet (richiede rete al primo uso).
    """
    weights = None
    if pretrained:
        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
        except Exception:  # pragma: no cover - offline / versione torchvision
            weights = None
    net = models.resnet18(weights=weights)
    in_features = net.fc.in_features
    net.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return net


def enable_mc_dropout(model: nn.Module) -> None:
    """Mette il modello in eval ma riattiva i layer Dropout (per MC-Dropout)."""
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
