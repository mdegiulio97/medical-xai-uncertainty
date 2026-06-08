"""Valuta il modello: metriche, calibrazione, baseline, incertezza e figure.

Se manca il checkpoint, addestra rapidamente. Salva in results/:
- metrics.json
- reliability_diagram.png
- gradcam_examples.png
- uncertainty_vs_correctness.png

Esegui:  uv run python scripts/evaluate.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from cxr_xai.baseline import logistic_baseline
from cxr_xai.calibration import (
    expected_calibration_error,
    fit_temperature,
    reliability_curve,
)
from cxr_xai.data import CLASS_NAMES, denormalize, load_split, loader_to_tensors
from cxr_xai.gradcam import GradCAM, resnet_target_layer
from cxr_xai.metrics import classification_metrics
from cxr_xai.model import build_model, get_device
from cxr_xai.training import collect_logits, probs_from_logits, train_model
from cxr_xai.uncertainty import mc_dropout_predict

RESULTS = Path(__file__).resolve().parents[1] / "results"
CKPT = RESULTS / "model.pt"


def _load_or_train(device):
    model = build_model(num_classes=2, dropout=0.3, pretrained=True)
    if CKPT.exists():
        model.load_state_dict(torch.load(CKPT, map_location=device)["state_dict"])
        print("Checkpoint caricato.")
    else:
        print("Nessun checkpoint: addestro 3 epoche...")
        tr = load_split("train")
        va = load_split("val")
        train_model(model, tr, va, epochs=3, device=device)
        RESULTS.mkdir(exist_ok=True)
        torch.save({"state_dict": model.state_dict()}, CKPT)
    model.to(device)
    return model


def reliability_diagram(prob_raw, prob_cal, y, path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfetta")
    for prob, name, mk in [(prob_raw, "non calibrato", "o-"), (prob_cal, "temp. scaling", "s-")]:
        xs, ys = reliability_curve(prob, y)
        ax.plot(xs, ys, mk, label=name)
    ax.set_xlabel("confidenza media")
    ax.set_ylabel("accuratezza")
    ax.set_title("Reliability diagram")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def gradcam_examples(model, x_test, y_test, prob, device, path, n=4):
    cam_fn = GradCAM(model, resnet_target_layer(model))
    idx = np.linspace(0, len(x_test) - 1, n).astype(int)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.2))
    for ax, i in zip(axes, idx):
        x = x_test[i : i + 1].to(device)
        cam = cam_fn(x)
        img = denormalize(x_test[i])
        ax.imshow(img, cmap="gray")
        ax.imshow(cam, cmap="jet", alpha=0.45)
        pred = CLASS_NAMES[int(prob[i] >= 0.5)]
        truth = CLASS_NAMES[int(y_test[i])]
        ax.set_title(f"pred={pred}\ntrue={truth} p={prob[i]:.2f}", fontsize=9)
        ax.axis("off")
    fig.suptitle("Grad-CAM su esempi di test")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def uncertainty_figure(epistemic, correct, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(epistemic[correct], bins=20, alpha=0.6, label="predizioni corrette", density=True)
    ax.hist(epistemic[~correct], bins=20, alpha=0.6, label="predizioni errate", density=True)
    ax.set_xlabel("incertezza epistemica (std MC-Dropout)")
    ax.set_ylabel("densità")
    ax.set_title("L'incertezza è più alta sugli errori")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    model = _load_or_train(device)

    val_loader = load_split("val", shuffle=False)
    test_loader = load_split("test", shuffle=False)
    x_test, y_test = loader_to_tensors(test_loader)
    y_np = y_test.numpy().ravel()

    # Logit di test + calibrazione su validation.
    test_logits, _ = collect_logits(model, test_loader, device)
    val_logits, val_labels = collect_logits(model, val_loader, device)
    temperature = fit_temperature(val_logits, val_labels)

    prob_raw = probs_from_logits(test_logits, temperature=1.0)
    prob_cal = probs_from_logits(test_logits, temperature=temperature)

    deep = classification_metrics(y_np, prob_cal)
    base = logistic_baseline(*loader_to_tensors(load_split("train", shuffle=False)), x_test, y_test)
    ece_raw = expected_calibration_error(prob_raw, y_np)
    ece_cal = expected_calibration_error(prob_cal, y_np)

    # Incertezza MC-Dropout.
    mean_prob, epistemic = mc_dropout_predict(model, x_test, n_samples=20, device=device)
    pred = (mean_prob[:, 1] >= 0.5).astype(int)
    correct = pred == y_np

    metrics = {
        "deep_model": deep,
        "logistic_baseline": base,
        "ece_uncalibrated": round(ece_raw, 4),
        "ece_temperature_scaled": round(ece_cal, 4),
        "temperature": round(temperature, 3),
        "mean_uncertainty_correct": round(float(epistemic[correct].mean()), 4),
        "mean_uncertainty_wrong": round(float(epistemic[~correct].mean()), 4) if (~correct).any() else None,
        "n_test": int(len(y_np)),
    }

    reliability_diagram(prob_raw, prob_cal, y_np, RESULTS / "reliability_diagram.png")
    gradcam_examples(model, x_test, y_test, prob_cal, device, RESULTS / "gradcam_examples.png")
    uncertainty_figure(epistemic, correct, RESULTS / "uncertainty_vs_correctness.png")
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print(f"\nSalvati metriche e 3 figure in {RESULTS}")


if __name__ == "__main__":
    main()
