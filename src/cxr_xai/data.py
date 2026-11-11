"""PneumoniaMNIST data loading (medmnist).

PneumoniaMNIST (MedMNIST v2) is a dataset of paediatric chest X-rays labelled
*pneumonia* vs *normal*, resized and standardised.
CC BY 4.0 licence, automatic download with no registration.

Reference: Yang J. et al., "MedMNIST v2 — A large-scale lightweight benchmark
for 2D and 3D biomedical image classification", Scientific Data 10:41, 2023.
DOI: 10.1038/s41597-022-01721-8
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

IMG_SIZE = 64
CLASS_NAMES = ("normal", "pneumonia")
DATA_ROOT = Path(__file__).resolve().parents[2] / "data"


def _transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )


def load_split(split: str, batch_size: int = 128, shuffle: bool | None = None) -> DataLoader:
    """Load a PneumoniaMNIST split as a DataLoader (3×64×64 tensors).

    `split` in {"train", "val", "test"}. Downloads the data to `data/` if missing.
    """
    from medmnist import PneumoniaMNIST  # local import: heavy and not needed in tests

    DATA_ROOT.mkdir(exist_ok=True)
    ds = PneumoniaMNIST(
        split=split, transform=_transform(), download=True, root=str(DATA_ROOT)
    )
    if shuffle is None:
        shuffle = split == "train"
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def loader_to_tensors(loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialise a DataLoader into (X, y), with y as 1-D integers."""
    xs, ys = [], []
    for x, y in loader:
        xs.append(x)
        ys.append(y)
    x = torch.cat(xs)
    y = torch.cat(ys).long().view(-1)
    return x, y


def make_fake_loader(
    n: int = 32, batch_size: int = 16, seed: int = 0
) -> DataLoader:
    """Synthetic (noise) DataLoader for fast tests without downloads.

    Creates two separable classes: class 1 has a higher mean intensity.
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 3, IMG_SIZE, IMG_SIZE, generator=g) * 0.1
    y = torch.randint(0, 2, (n,), generator=g)
    x += y.view(-1, 1, 1, 1).float() * 0.5  # weak signal correlated with the class
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)


def denormalize(img: torch.Tensor) -> np.ndarray:
    """Convert a normalised tensor (3×H×W) back to a [0,1] image (H×W)."""
    arr = img.detach().cpu().numpy()
    arr = arr * 0.5 + 0.5
    return np.clip(arr[0], 0.0, 1.0)
