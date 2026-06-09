"""Caricamento del dataset Cheng di tumori cerebrali (MRI T1w CE) per il track reale.

Cheng et al. (figshare, CC BY 4.0): 3064 slice T1w con contrasto, etichettate in
3 classi (1 = meningioma, 2 = glioma, 3 = adenoma ipofisario), in file MATLAB
v7.3 (formato HDF5). Ogni file contiene una struct ``cjdata`` con ``image`` e
``label``.

Qui si imposta un task **binario clinicamente rilevante**: glioma (la classe
maligna) vs non-glioma. Le immagini sono normalizzate e portate nel formato della
pipeline (3×64×64, normalizzazione ±1), così da riusare modello, Grad-CAM,
incertezza e calibrazione del track chest senza modifiche.

Riferimento: Cheng J. et al., brain tumor dataset, figshare,
DOI: 10.6084/m9.figshare.1512427 (CC BY 4.0).
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

IMG_SIZE = 64
GLIOMA_LABEL = 2
CLASS_NAMES = ("non-glioma", "glioma")


def read_mat_image(path: Path | str) -> tuple[np.ndarray, int]:
    """Legge un file .mat v7.3 di Cheng: ritorna (immagine HxW float, label int)."""
    with h5py.File(str(path), "r") as f:
        cj = f["cjdata"]
        label = int(np.asarray(cj["label"]).ravel()[0])
        image = np.asarray(cj["image"], dtype=np.float32)
    return image, label


def image_to_tensor(image: np.ndarray, img_size: int = IMG_SIZE) -> torch.Tensor:
    """Normalizza in [0,1], ridimensiona a img_size, 3 canali, normalizza a ±1.

    Stesso formato del track chest: (3, img_size, img_size), Normalize(0.5, 0.5).
    """
    img = image.astype(np.float32)
    lo, hi = float(img.min()), float(img.max())
    img = (img - lo) / max(hi - lo, 1e-6)
    t = torch.from_numpy(img)[None, None]  # (1,1,H,W)
    t = F.interpolate(t, size=(img_size, img_size), mode="bilinear", align_corners=False)
    t = t.repeat(1, 3, 1, 1)  # 3 canali
    t = (t - 0.5) / 0.5
    return t[0]  # (3, img_size, img_size)


def load_tumor_tensors(
    mat_dir: Path | str,
    img_size: int = IMG_SIZE,
    positive_label: int = GLIOMA_LABEL,
    max_n: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Carica tutti i .mat in una directory come (X (N,3,H,W), y (N,)) binari.

    y = 1 se la classe è ``positive_label`` (glioma), altrimenti 0.
    """
    files = sorted(Path(mat_dir).glob("*.mat"))
    if not files:
        raise FileNotFoundError(f"Nessun .mat in {mat_dir}")
    if max_n is not None:
        files = files[:max_n]
    xs, ys = [], []
    for p in files:
        image, label = read_mat_image(p)
        xs.append(image_to_tensor(image, img_size))
        ys.append(1 if label == positive_label else 0)
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long)


def make_loaders(
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int = 64,
    seed: int = 0,
    fracs: tuple[float, float, float] = (0.7, 0.1, 0.2),
) -> dict[str, DataLoader]:
    """Split casuale (riproducibile) in train/val/test come DataLoader."""
    n = x.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    n_tr = int(fracs[0] * n)
    n_va = int(fracs[1] * n)
    splits = {
        "train": perm[:n_tr],
        "val": perm[n_tr : n_tr + n_va],
        "test": perm[n_tr + n_va :],
    }
    loaders = {}
    for name, idx in splits.items():
        ds = TensorDataset(x[idx], y[idx])
        loaders[name] = DataLoader(ds, batch_size=batch_size, shuffle=(name == "train"))
    return loaders
