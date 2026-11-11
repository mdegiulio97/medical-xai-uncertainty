"""Train the classifier and save the checkpoint to results/model.pt.

Run:  uv run python scripts/train.py --epochs 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cxr_xai.data import load_split
from cxr_xai.model import build_model, get_device
from cxr_xai.training import train_model

RESULTS = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--no-pretrained", action="store_true")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    train_loader = load_split("train", batch_size=args.batch_size)
    val_loader = load_split("val", batch_size=args.batch_size)

    model = build_model(num_classes=2, dropout=0.3, pretrained=not args.no_pretrained)
    history = train_model(model, train_loader, val_loader, epochs=args.epochs, lr=args.lr, device=device)

    ckpt = RESULTS / "model.pt"
    torch.save({"state_dict": model.state_dict(), "history": history}, ckpt)
    print(f"Saved checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
