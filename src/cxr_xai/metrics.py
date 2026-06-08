"""Metriche di classificazione binaria."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def classification_metrics(y_true: np.ndarray, prob_pos: np.ndarray) -> dict:
    """AUROC, F1 e accuracy data la probabilità della classe positiva (pneumonia)."""
    y_true = np.asarray(y_true).ravel()
    prob_pos = np.asarray(prob_pos).ravel()
    pred = (prob_pos >= 0.5).astype(int)
    return {
        "auroc": float(roc_auc_score(y_true, prob_pos)),
        "f1": float(f1_score(y_true, pred)),
        "accuracy": float((pred == y_true).mean()),
    }
