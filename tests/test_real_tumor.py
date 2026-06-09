"""Test offline del loader Cheng (nessun download).

Si scrivono alcuni file .mat v7.3 (HDF5) sintetici con struct ``cjdata`` e si
verifica lettura, etichettatura binaria (glioma vs non-glioma), formato dei
tensori e split in DataLoader.
"""

from __future__ import annotations

import h5py
import numpy as np
import torch

from cxr_xai.real_tumor import (
    image_to_tensor,
    load_tumor_tensors,
    make_loaders,
    read_mat_image,
)


def _write_mat(path, image, label):
    with h5py.File(str(path), "w") as f:
        g = f.create_group("cjdata")
        g.create_dataset("image", data=image.astype(np.float32))
        g.create_dataset("label", data=np.array([[float(label)]]))


def test_read_mat_image(tmp_path):
    img = np.arange(20 * 24, dtype=np.float32).reshape(20, 24)
    _write_mat(tmp_path / "1.mat", img, 2)
    out_img, label = read_mat_image(tmp_path / "1.mat")
    assert label == 2
    assert out_img.shape == (20, 24)


def test_image_to_tensor_format():
    img = np.random.default_rng(0).uniform(0, 4000, size=(50, 60)).astype(np.float32)
    t = image_to_tensor(img, img_size=64)
    assert t.shape == (3, 64, 64)
    assert float(t.min()) >= -1.01 and float(t.max()) <= 1.01


def test_load_tumor_tensors_binary_labels(tmp_path):
    rng = np.random.default_rng(0)
    for i, label in enumerate([1, 2, 3, 2], start=1):
        _write_mat(tmp_path / f"{i}.mat", rng.uniform(0, 1000, size=(40, 40)), label)
    x, y = load_tumor_tensors(tmp_path, img_size=32)
    assert x.shape == (4, 3, 32, 32)
    # glioma = label 2 -> 1; gli altri -> 0
    assert y.tolist() == [0, 1, 0, 1]


def test_make_loaders_splits(tmp_path):
    x = torch.randn(20, 3, 32, 32)
    y = torch.randint(0, 2, (20,))
    loaders = make_loaders(x, y, batch_size=4, seed=0)
    total = sum(len(dl.dataset) for dl in loaders.values())
    assert total == 20
    assert set(loaders) == {"train", "val", "test"}
    assert len(loaders["train"].dataset) > len(loaders["test"].dataset)
