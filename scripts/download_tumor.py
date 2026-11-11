"""Download and extract the full Cheng brain-tumour dataset (figshare, CC BY 4.0).

All four public parts (3064 contrast-enhanced T1w slices, 233 patients) are
downloaded, checked against the figshare MD5 and extracted into
data/tumor/images/. Idempotent. Raw data are NOT committed (data/ is ignored).

Reference: Cheng J. et al., brain tumor dataset, figshare,
DOI 10.6084/m9.figshare.1512427 (CC BY 4.0).

Run:  uv run python scripts/download_tumor.py
"""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import requests

PARTS = [
    {"id": 3381290, "name": "brainTumorDataPublic_1-766.zip", "md5": "74b949ad33f042e6e103523091cd1428"},
    {"id": 3381296, "name": "brainTumorDataPublic_767-1532.zip", "md5": "7e8a875500d2c8a346f270538e29890e"},
    {"id": 3381293, "name": "brainTumorDataPublic_1533-2298.zip", "md5": "8227bf6080cb71f15a88be8d25c79ae7"},
    {"id": 3381302, "name": "brainTumorDataPublic_2299-3064.zip", "md5": "b378a80d6174e5317d59eb28430c6652"},
]
DEST = Path(__file__).resolve().parents[1] / "data" / "tumor"
IMAGES_DIR = DEST / "images"
N_SLICES = 3064


def md5_ok(path: Path, expected: str) -> bool:
    """True if `path` exists and its MD5 equals `expected`."""
    if not path.exists():
        return False
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest() == expected


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(exist_ok=True)
    print("Dataset: Cheng brain tumor (figshare) - CC BY 4.0")
    for part in PARTS:
        zip_path = DEST / part["name"]
        if md5_ok(zip_path, part["md5"]):
            print(f"[skip] {part['name']} already present (MD5 ok)")
        else:
            print(f"Downloading {part['name']}...")
            url = f"https://ndownloader.figshare.com/files/{part['id']}"
            with requests.get(url, stream=True, timeout=600) as r:
                r.raise_for_status()
                with zip_path.open("wb") as fh:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
            if not md5_ok(zip_path, part["md5"]):
                raise SystemExit(f"MD5 mismatch for {part['name']}: delete it and retry")
        with zipfile.ZipFile(zip_path) as zf:
            for name in zf.namelist():
                out = IMAGES_DIR / Path(name).name
                if name.endswith(".mat") and not out.exists():
                    out.write_bytes(zf.read(name))
    n = len(list(IMAGES_DIR.glob("*.mat")))
    print(f"{n} slices ready in {IMAGES_DIR}")
    if n != N_SLICES:
        raise SystemExit(f"Expected {N_SLICES} slices, found {n}")
    print("Citation: Cheng et al., DOI 10.6084/m9.figshare.1512427 (CC BY 4.0).")


if __name__ == "__main__":
    main()
