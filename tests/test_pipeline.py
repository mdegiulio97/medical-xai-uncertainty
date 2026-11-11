"""Fast pipeline tests (no download, backbone not pre-trained)."""

import numpy as np
import torch

from cxr_xai.baseline import logistic_baseline
from cxr_xai.calibration import expected_calibration_error, fit_temperature, reliability_curve
from cxr_xai.data import loader_to_tensors, make_fake_loader
from cxr_xai.gradcam import GradCAM, resnet_target_layer
from cxr_xai.metrics import classification_metrics
from cxr_xai.model import build_model, enable_mc_dropout
from cxr_xai.training import collect_logits, probs_from_logits, train_model
from cxr_xai.uncertainty import mc_dropout_predict


def _model():
    torch.manual_seed(0)
    return build_model(num_classes=2, dropout=0.3, pretrained=False)


def test_model_forward_shape():
    model = _model()
    x = torch.randn(4, 3, 64, 64)
    out = model(x)
    assert out.shape == (4, 2)


def test_enable_mc_dropout_keeps_dropout_training():
    model = _model()
    enable_mc_dropout(model)
    dropouts = [m for m in model.modules() if isinstance(m, torch.nn.Dropout)]
    assert dropouts and all(m.training for m in dropouts)


def test_mc_dropout_predict_shapes():
    model = _model()
    x = torch.randn(5, 3, 64, 64)
    mean_prob, epistemic = mc_dropout_predict(model, x, n_samples=8)
    assert mean_prob.shape == (5, 2)
    assert epistemic.shape == (5,)
    assert np.allclose(mean_prob.sum(axis=1), 1.0, atol=1e-5)


def test_gradcam_shape_and_range():
    model = _model()
    cam_fn = GradCAM(model, resnet_target_layer(model))
    x = torch.randn(1, 3, 64, 64)
    cam = cam_fn(x)
    assert cam.shape == (64, 64)
    assert cam.min() >= 0.0 and cam.max() <= 1.0


def test_metrics_on_separable_scores():
    y = np.array([0, 0, 1, 1])
    prob = np.array([0.1, 0.2, 0.8, 0.9])
    m = classification_metrics(y, prob)
    assert m["auroc"] == 1.0 and m["accuracy"] == 1.0


def test_calibration_helpers():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=200)
    prob = np.clip(y * 0.6 + rng.normal(0.2, 0.2, size=200), 0.01, 0.99)
    ece = expected_calibration_error(prob, y)
    assert 0.0 <= ece <= 1.0
    xs, ys = reliability_curve(prob, y)
    assert len(xs) == len(ys)

    logits = torch.tensor(np.stack([1 - prob, prob], axis=1) * 3.0, dtype=torch.float32)
    t = fit_temperature(logits, torch.tensor(y))
    assert t > 0.0


def test_train_reduces_loss():
    model = _model()
    loader = make_fake_loader(n=64, batch_size=16, seed=1)
    history = train_model(model, loader, epochs=3, lr=1e-3, verbose=False)
    assert history["train_loss"][-1] < history["train_loss"][0]


def test_logistic_baseline_runs():
    tr = make_fake_loader(n=64, seed=2)
    te = make_fake_loader(n=32, seed=3)
    xtr, ytr = loader_to_tensors(tr)
    xte, yte = loader_to_tensors(te)
    m = logistic_baseline(xtr, ytr, xte, yte)
    assert set(m) == {"auroc", "f1", "accuracy"}


def test_collect_logits_and_probs():
    model = _model()
    loader = make_fake_loader(n=32, seed=4)
    logits, labels = collect_logits(model, loader)
    assert logits.shape[0] == labels.shape[0]
    prob = probs_from_logits(logits, temperature=1.5)
    assert prob.shape[0] == logits.shape[0]
