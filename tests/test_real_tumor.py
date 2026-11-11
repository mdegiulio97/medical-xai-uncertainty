"""Offline tests of the Cheng loader (no download).

A few synthetic .mat v7.3 (HDF5) files with a ``cjdata`` struct are written, then
reading, binary labelling (glioma vs non-glioma), tensor format and the
DataLoader split are checked.
"""

from __future__ import annotations

import h5py
import numpy as np
import torch

from cxr_xai.real_tumor import (
    group_split_indices,
    image_to_tensor,
    load_tumor_tensors,
    make_loaders,
    read_mat_image,
    read_mat_pid,
)


def _write_mat(path, image, label, pid="100"):
    with h5py.File(str(path), "w") as f:
        g = f.create_group("cjdata")
        g.create_dataset("image", data=image.astype(np.float32))
        g.create_dataset("label", data=np.array([[float(label)]]))
        # MATLAB v7.3 stores char arrays as uint16 character codes
        g.create_dataset("PID", data=np.array([[ord(c)] for c in pid], dtype=np.uint16))


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
    x, y, groups = load_tumor_tensors(tmp_path, img_size=32)
    assert x.shape == (4, 3, 32, 32)
    assert len(groups) == 4
    # glioma = label 2 -> 1; all others -> 0
    assert y.tolist() == [0, 1, 0, 1]


def _grouped_data(n_patients=40, slices=3):
    groups = np.repeat([f"P{i:03d}" for i in range(n_patients)], slices).astype(object)
    y = np.repeat([i % 2 for i in range(n_patients)], slices)
    return y, groups


def test_make_loaders_splits():
    y_np, groups = _grouped_data()
    x = torch.randn(len(y_np), 3, 32, 32)
    y = torch.tensor(y_np)
    loaders = make_loaders(x, y, groups, batch_size=4, seed=0)
    total = sum(len(dl.dataset) for dl in loaders.values())
    assert total == len(y_np)
    assert set(loaders) == {"train", "val", "test"}
    assert len(loaders["train"].dataset) > len(loaders["test"].dataset)


def test_read_mat_pid(tmp_path):
    _write_mat(tmp_path / "1.mat", np.zeros((8, 8)), 2, pid="112233")
    assert read_mat_pid(tmp_path / "1.mat") == "112233"


def test_group_split_no_patient_in_two_splits():
    y, groups = _grouped_data()
    idx = group_split_indices(y, groups, seed=0)
    sets = {k: set(groups[v]) for k, v in idx.items()}
    assert sets["train"].isdisjoint(sets["val"])
    assert sets["train"].isdisjoint(sets["test"])
    assert sets["val"].isdisjoint(sets["test"])
    assert sorted(np.concatenate(list(idx.values())).tolist()) == list(range(len(y)))


def test_group_split_each_split_has_both_classes():
    y, groups = _grouped_data()
    idx = group_split_indices(y, groups, seed=0)
    for name, v in idx.items():
        assert set(y[v].tolist()) == {0, 1}, name


def test_group_split_reproducible():
    y, groups = _grouped_data()
    a = group_split_indices(y, groups, seed=3)
    b = group_split_indices(y, groups, seed=3)
    for k in a:
        assert np.array_equal(a[k], b[k])


def test_patient_group_strips_series_suffix():
    from cxr_xai.real_tumor import patient_group

    assert patient_group("MR040240B") == "MR040240"
    assert patient_group("MR040240") == "MR040240"
    assert patient_group("MR029209E") == "MR029209"
    assert patient_group("100360") == "100360"


def test_load_groups_series_of_same_patient(tmp_path):
    rng = np.random.default_rng(0)
    for i, pid in enumerate(["MR051644", "MR051644B", "MR051644C", "112233"], start=1):
        _write_mat(tmp_path / f"{i}.mat", rng.uniform(0, 1000, size=(16, 16)), 2, pid=pid)
    _, _, groups = load_tumor_tensors(tmp_path, img_size=16)
    assert sorted(set(groups)) == ["112233", "MR051644"]
