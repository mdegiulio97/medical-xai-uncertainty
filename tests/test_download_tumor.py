"""Offline checks for the Cheng download helper (no network)."""

import hashlib
import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "download_tumor", Path(__file__).resolve().parents[1] / "scripts" / "download_tumor.py"
)
dl = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(dl)


def test_all_four_parts_listed():
    assert len(dl.PARTS) == 4
    assert {p["id"] for p in dl.PARTS} == {3381290, 3381296, 3381293, 3381302}


def test_md5_ok(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"cheng")
    assert dl.md5_ok(f, hashlib.md5(b"cheng").hexdigest())
    assert not dl.md5_ok(f, "0" * 32)
    assert not dl.md5_ok(tmp_path / "missing.bin", "0" * 32)
