"""Uncertainty quantification with MC-Dropout.

T forward passes are run with the Dropout layers active: the mean of the
probabilities is the prediction, and the standard deviation across samples
estimates the epistemic (model) uncertainty.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from cxr_xai.model import enable_mc_dropout


@torch.no_grad()
def mc_dropout_predict(
    model: torch.nn.Module,
    x: torch.Tensor,
    n_samples: int = 20,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """MC-Dropout prediction.

    Returns
    -------
    mean_prob : (N, C) probabilities averaged over the T iterations.
    epistemic : (N,) standard deviation of the positive-class probability.
    """
    device = device or next(model.parameters()).device
    enable_mc_dropout(model)
    x = x.to(device)

    probs = []
    for _ in range(n_samples):
        logits = model(x)
        probs.append(F.softmax(logits, dim=1).cpu().numpy())
    probs = np.stack(probs, axis=0)  # (T, N, C)

    mean_prob = probs.mean(axis=0)  # (N, C)
    epistemic = probs[:, :, 1].std(axis=0)  # (N,)
    return mean_prob, epistemic
