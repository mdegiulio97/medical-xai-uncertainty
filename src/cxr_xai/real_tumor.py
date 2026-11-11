"""Loader for the Cheng brain-tumour dataset (contrast-enhanced T1w MRI), real-data track.

Cheng et al. (figshare, CC BY 4.0): 3064 contrast-enhanced T1w slices from 233
patients, labelled in 3 classes (1 = meningioma, 2 = glioma, 3 = pituitary
tumour), stored as MATLAB v7.3 (HDF5) files. Each file holds a ``cjdata`` struct
with ``image``, ``label`` and ``PID`` (patient ID).

The task is a clinically relevant **binary** one: glioma (the malignant class)
vs non-glioma. Images are normalised and brought to the pipeline format
(3x64x64, +/-1 normalisation), so that model, Grad-CAM, uncertainty and
calibration from the chest track are reused unchanged.

Split is by patient (PID); no patient appears in more than one split. Slices of
the same patient are highly correlated, so a slice-level split would leak.

Reference: Cheng J. et al., brain tumor dataset, figshare,
DOI: 10.6084/m9.figshare.1512427 (CC BY 4.0).
"""

from __future__ import annotations

import re
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, TensorDataset

IMG_SIZE = 64
GLIOMA_LABEL = 2
CLASS_NAMES = ("non-glioma", "glioma")


def read_mat_image(path: Path | str) -> tuple[np.ndarray, int]:
    """Read a Cheng v7.3 .mat file: return (HxW float image, int label)."""
    with h5py.File(str(path), "r") as f:
        cj = f["cjdata"]
        label = int(np.asarray(cj["label"]).ravel()[0])
        image = np.asarray(cj["image"], dtype=np.float32)
    return image, label


def read_mat_pid(path: Path | str) -> str:
    """Read the patient ID (``cjdata/PID``, stored as character codes)."""
    with h5py.File(str(path), "r") as f:
        codes = np.asarray(f["cjdata"]["PID"]).ravel()
    return "".join(chr(int(c)) for c in codes).strip()


_SERIES_SUFFIX = re.compile(r"^(MR\d+)[A-Z]$")


def patient_group(pid: str) -> str:
    """Grouping key for a PID: MR IDs that differ only by a trailing letter
    (e.g. MR040240, MR040240B, MR040240C) are treated as the same patient."""
    m = _SERIES_SUFFIX.match(pid)
    return m.group(1) if m else pid


def image_to_tensor(image: np.ndarray, img_size: int = IMG_SIZE) -> torch.Tensor:
    """Scale to [0,1], resize to img_size, 3 channels, normalise to +/-1.

    Same format as the chest track: (3, img_size, img_size), Normalize(0.5, 0.5).
    """
    img = image.astype(np.float32)
    lo, hi = float(img.min()), float(img.max())
    img = (img - lo) / max(hi - lo, 1e-6)
    t = torch.from_numpy(img)[None, None]  # (1,1,H,W)
    t = F.interpolate(t, size=(img_size, img_size), mode="bilinear", align_corners=False)
    t = t.repeat(1, 3, 1, 1)  # 3 channels
    t = (t - 0.5) / 0.5
    return t[0]  # (3, img_size, img_size)


def load_tumor_tensors(
    mat_dir: Path | str,
    img_size: int = IMG_SIZE,
    positive_label: int = GLIOMA_LABEL,
    max_n: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Load every .mat in a directory as binary (X (N,3,H,W), y (N,), groups (N,)).

    y = 1 if the class is ``positive_label`` (glioma), else 0. ``groups`` holds
    the patient group of each slice (see ``patient_group``), used for
    patient-level splitting.
    """
    files = sorted(Path(mat_dir).glob("*.mat"))
    if not files:
        raise FileNotFoundError(f"No .mat files in {mat_dir}")
    if max_n is not None:
        files = files[:max_n]
    xs, ys, pids = [], [], []
    for p in files:
        image, label = read_mat_image(p)
        xs.append(image_to_tensor(image, img_size))
        ys.append(1 if label == positive_label else 0)
        pids.append(patient_group(read_mat_pid(p)))
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long), np.array(pids, dtype=object)


def group_split_indices(
    y: np.ndarray,
    groups: np.ndarray,
    fracs: tuple[float, float, float] = (0.7, 0.1, 0.2),
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Patient-level, class-stratified train/val/test split (indices).

    Test = first fold of a StratifiedGroupKFold with ~1/fracs[2] folds; val =
    first fold of the same on the remainder; the rest is train. No group is
    shared between splits.
    """
    y = np.asarray(y).ravel()
    groups = np.asarray(groups)
    idx = np.arange(len(y))
    n_test = max(2, round(1 / fracs[2]))
    sgkf = StratifiedGroupKFold(n_splits=n_test, shuffle=True, random_state=seed)
    rest, test = next(sgkf.split(idx, y, groups))
    n_val = max(2, round((1 - fracs[2]) / fracs[1]))
    sgkf = StratifiedGroupKFold(n_splits=n_val, shuffle=True, random_state=seed)
    tr, va = next(sgkf.split(rest, y[rest], groups[rest]))
    return {"train": np.sort(rest[tr]), "val": np.sort(rest[va]), "test": np.sort(test)}


def make_loaders(
    x: torch.Tensor,
    y: torch.Tensor,
    groups: np.ndarray,
    batch_size: int = 64,
    seed: int = 0,
    fracs: tuple[float, float, float] = (0.7, 0.1, 0.2),
) -> dict[str, DataLoader]:
    """Patient-level train/val/test split (reproducible) as DataLoaders."""
    splits = group_split_indices(y.numpy(), groups, fracs=fracs, seed=seed)
    loaders = {}
    for name, idx in splits.items():
        t = torch.from_numpy(idx)
        ds = TensorDataset(x[t], y[t])
        loaders[name] = DataLoader(ds, batch_size=batch_size, shuffle=(name == "train"))
    return loaders
