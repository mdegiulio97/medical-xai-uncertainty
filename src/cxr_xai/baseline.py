"""Baseline: logistic regression on raw pixels.

A simple, non-deep reference: flattens the images and trains a logistic
regression. It measures how much the deep model adds (AUROC/F1) over a
linear classifier.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from cxr_xai.metrics import classification_metrics


def _flatten(x: torch.Tensor) -> np.ndarray:
    # use a single channel (the images are greyscale replicated across channels)
    return x[:, 0].reshape(x.shape[0], -1).cpu().numpy()


def logistic_baseline(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
) -> dict:
    """Train the logistic regression and return the test-set metrics."""
    xtr = _flatten(x_train)
    xte = _flatten(x_test)
    ytr = y_train.cpu().numpy().ravel()
    yte = y_test.cpu().numpy().ravel()

    clf = LogisticRegression(max_iter=200, C=1.0)
    clf.fit(xtr, ytr)
    prob = clf.predict_proba(xte)[:, 1]
    return classification_metrics(yte, prob)
