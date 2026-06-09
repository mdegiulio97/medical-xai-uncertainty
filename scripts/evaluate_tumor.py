"""Track REALE: classificazione tumore cerebrale (glioma vs non-glioma) su MRI Cheng.

Riusa l'intera metodologia del track chest — ResNet18, Grad-CAM, incertezza
MC-Dropout, calibrazione con temperature scaling, baseline logistica — applicata a
slice T1w cerebrali reali (Cheng, CC BY 4.0). Senza modifiche al modello.

Produce:
- results/tumor_metrics.json
- results/tumor_reliability.png
- results/tumor_gradcam.png
- results/tumor_uncertainty.png

Uso:
    uv run python scripts/download_tumor.py
    uv run python scripts/evaluate_tumor.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from cxr_xai.baseline import logistic_baseline  # noqa: E402
from cxr_xai.calibration import (  # noqa: E402
    expected_calibration_error,
    fit_temperature,
    reliability_curve,
)
from cxr_xai.gradcam import GradCAM, resnet_target_layer  # noqa: E402
from cxr_xai.metrics import classification_metrics  # noqa: E402
from cxr_xai.model import build_model, get_device  # noqa: E402
from cxr_xai.real_tumor import CLASS_NAMES, IMG_SIZE, load_tumor_tensors, make_loaders  # noqa: E402
from cxr_xai.training import collect_logits, probs_from_logits, train_model  # noqa: E402
from cxr_xai.uncertainty import mc_dropout_predict  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
IMAGES_DIR = ROOT / "data" / "tumor" / "images"
RESULTS = ROOT / "results"


def _denorm(img: torch.Tensor) -> np.ndarray:
    arr = img.detach().cpu().numpy() * 0.5 + 0.5
    return np.clip(arr[0], 0.0, 1.0)


def _tensors(loader):
    xs, ys = [], []
    for x, y in loader:
        xs.append(x)
        ys.append(y)
    return torch.cat(xs), torch.cat(ys).long().view(-1)


def main() -> None:
    if not IMAGES_DIR.exists() or not list(IMAGES_DIR.glob("*.mat")):
        raise SystemExit("Dati assenti. Esegui prima:  uv run python scripts/download_tumor.py")

    RESULTS.mkdir(exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    x, y = load_tumor_tensors(IMAGES_DIR, img_size=IMG_SIZE)
    pos = int(y.sum())
    print(f"{len(y)} slice — glioma={pos}, non-glioma={len(y) - pos}")
    loaders = make_loaders(x, y, batch_size=64, seed=0)

    model = build_model(num_classes=2, dropout=0.3, pretrained=True)
    print("Addestro ResNet18 (5 epoche)...")
    train_model(model, loaders["train"], loaders["val"], epochs=5, device=device)
    model.to(device)

    x_test, y_test = _tensors(loaders["test"])
    y_np = y_test.numpy().ravel()

    test_logits, _ = collect_logits(model, loaders["test"], device)
    val_logits, val_labels = collect_logits(model, loaders["val"], device)
    temperature = fit_temperature(val_logits, val_labels)
    prob_raw = probs_from_logits(test_logits, temperature=1.0)
    prob_cal = probs_from_logits(test_logits, temperature=temperature)

    deep = classification_metrics(y_np, prob_cal)
    x_tr, y_tr = _tensors(loaders["train"])
    base = logistic_baseline(x_tr, y_tr, x_test, y_test)
    ece_raw = expected_calibration_error(prob_raw, y_np)
    ece_cal = expected_calibration_error(prob_cal, y_np)

    mean_prob, epistemic = mc_dropout_predict(model, x_test, n_samples=20, device=device)
    pred = (mean_prob[:, 1] >= 0.5).astype(int)
    correct = pred == y_np

    metrics = {
        "dataset": {
            "name": "Cheng brain tumor (figshare)",
            "license": "CC BY 4.0",
            "doi": "10.6084/m9.figshare.1512427",
            "task": "glioma vs non-glioma (binario)",
            "modality": "T1w CE (brain MRI)",
        },
        "deep_model": deep,
        "logistic_baseline": base,
        "ece_uncalibrated": round(ece_raw, 4),
        "ece_temperature_scaled": round(ece_cal, 4),
        "temperature": round(temperature, 3),
        "mean_uncertainty_correct": round(float(epistemic[correct].mean()), 4),
        "mean_uncertainty_wrong": (
            round(float(epistemic[~correct].mean()), 4) if (~correct).any() else None
        ),
        "n_test": int(len(y_np)),
    }
    (RESULTS / "tumor_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nAUROC deep={deep['auroc']:.3f}  baseline={base['auroc']:.3f}  "
          f"ECE {ece_raw:.3f}->{ece_cal:.3f}")
    print(f"Incertezza media: corretti={metrics['mean_uncertainty_correct']}  "
          f"errati={metrics['mean_uncertainty_wrong']}")

    # --- Reliability diagram ---
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfetta")
    for prob, name, mk in [(prob_raw, "non calibrato", "o-"), (prob_cal, "temp. scaling", "s-")]:
        xs, ys = reliability_curve(prob, y_np)
        ax.plot(xs, ys, mk, label=name)
    ax.set_xlabel("confidenza media")
    ax.set_ylabel("accuratezza")
    ax.set_title("Reliability diagram — glioma vs non-glioma")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "tumor_reliability.png", dpi=130)
    plt.close(fig)

    # --- Grad-CAM ---
    cam_fn = GradCAM(model, resnet_target_layer(model))
    idx = np.linspace(0, len(x_test) - 1, 4).astype(int)
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
    for ax, i in zip(axes, idx):
        cam = cam_fn(x_test[i : i + 1].to(device))
        ax.imshow(_denorm(x_test[i]), cmap="gray")
        ax.imshow(cam, cmap="jet", alpha=0.45)
        ax.set_title(
            f"pred={CLASS_NAMES[int(prob_cal[i] >= 0.5)]}\n"
            f"true={CLASS_NAMES[int(y_test[i])]} p={prob_cal[i]:.2f}",
            fontsize=9,
        )
        ax.axis("off")
    fig.suptitle("Grad-CAM su MRI cerebrale (Cheng, CC BY 4.0)")
    fig.tight_layout()
    fig.savefig(RESULTS / "tumor_gradcam.png", dpi=130)
    plt.close(fig)

    # --- Incertezza ---
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(epistemic[correct], bins=20, alpha=0.6, label="corrette", density=True)
    if (~correct).any():
        ax.hist(epistemic[~correct], bins=20, alpha=0.6, label="errate", density=True)
    ax.set_xlabel("incertezza epistemica (std MC-Dropout)")
    ax.set_ylabel("densità")
    ax.set_title("Incertezza più alta sugli errori")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "tumor_uncertainty.png", dpi=130)
    plt.close(fig)

    print(f"Salvati: tumor_metrics.json + 3 figure in {RESULTS}")


if __name__ == "__main__":
    main()
