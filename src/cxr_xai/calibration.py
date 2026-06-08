"""Calibrazione della confidenza: temperature scaling + Expected Calibration Error.

La temperatura T (scalare) riscala i logit (logit/T) per allineare la confidenza
predetta all'accuratezza reale. L'ECE misura il disallineamento medio in bin di
confidenza.

Riferimento: Guo C. et al., "On Calibration of Modern Neural Networks", ICML 2017.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def expected_calibration_error(prob_pos: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """ECE per classificazione binaria sulla confidenza della classe predetta."""
    prob_pos = np.asarray(prob_pos).ravel()
    y_true = np.asarray(y_true).ravel()
    conf = np.where(prob_pos >= 0.5, prob_pos, 1 - prob_pos)
    pred = (prob_pos >= 0.5).astype(int)
    correct = (pred == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(conf)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        ece += mask.sum() / n * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 100) -> float:
    """Stima la temperatura ottimale minimizzando la NLL (LBFGS). Ritorna T > 0."""
    logits = logits.detach()
    labels = labels.detach().long().view(-1)
    log_t = torch.zeros(1, requires_grad=True)  # T = exp(log_t) > 0
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.exp().item())


def reliability_curve(prob_pos: np.ndarray, y_true: np.ndarray, n_bins: int = 10):
    """Punti (confidenza media, accuratezza) per il reliability diagram."""
    prob_pos = np.asarray(prob_pos).ravel()
    y_true = np.asarray(y_true).ravel()
    conf = np.where(prob_pos >= 0.5, prob_pos, 1 - prob_pos)
    pred = (prob_pos >= 0.5).astype(int)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    xs, ys = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        xs.append(conf[mask].mean())
        ys.append(correct[mask].mean())
    return np.array(xs), np.array(ys)
