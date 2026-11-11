"""Training loop and logit collection for calibration."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    epochs: int = 3,
    lr: float = 1e-3,
    device: torch.device | None = None,
    verbose: bool = True,
) -> dict:
    """Train the model (Adam + cross-entropy). Returns the loss history."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device).long().view(-1)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
        train_loss = running / len(train_loader.dataset)
        history["train_loss"].append(train_loss)

        val_loss = float("nan")
        if val_loader is not None:
            val_loss = _eval_loss(model, val_loader, device)
            history["val_loss"].append(val_loss)
        if verbose:
            print(f"epoch {epoch + 1}/{epochs} | train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
    return history


@torch.no_grad()
def _eval_loss(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device).long().view(-1)
        total += F.cross_entropy(model(x), y, reduction="sum").item()
    return total / len(loader.dataset)


@torch.no_grad()
def collect_logits(
    model: nn.Module, loader: DataLoader, device: torch.device | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect (logits, labels) over a loader, in eval mode (dropout off)."""
    device = device or next(model.parameters()).device
    model.eval()
    logits, labels = [], []
    for x, y in loader:
        logits.append(model(x.to(device)).cpu())
        labels.append(y.long().view(-1))
    return torch.cat(logits), torch.cat(labels)


def probs_from_logits(logits: torch.Tensor, temperature: float = 1.0) -> np.ndarray:
    """Positive-class probability from logits, with optional temperature."""
    p = F.softmax(logits / temperature, dim=1).numpy()
    return p[:, 1]
