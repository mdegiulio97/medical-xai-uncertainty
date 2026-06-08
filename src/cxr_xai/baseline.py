"""Baseline: regressione logistica sui pixel grezzi.

Riferimento di confronto semplice e non-deep: appiattisce le immagini e addestra
una regressione logistica. Serve a quantificare quanto valore aggiunge il
modello profondo (AUROC/F1) rispetto a un classificatore lineare.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from cxr_xai.metrics import classification_metrics


def _flatten(x: torch.Tensor) -> np.ndarray:
    # usa un solo canale (le immagini sono in scala di grigi replicata)
    return x[:, 0].reshape(x.shape[0], -1).cpu().numpy()


def logistic_baseline(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
) -> dict:
    """Addestra la regressione logistica e ritorna le metriche sul test."""
    xtr = _flatten(x_train)
    xte = _flatten(x_test)
    ytr = y_train.cpu().numpy().ravel()
    yte = y_test.cpu().numpy().ravel()

    clf = LogisticRegression(max_iter=200, C=1.0)
    clf.fit(xtr, ytr)
    prob = clf.predict_proba(xte)[:, 1]
    return classification_metrics(yte, prob)
