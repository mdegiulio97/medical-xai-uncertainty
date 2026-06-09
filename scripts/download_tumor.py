"""Scarica ed estrae il dataset Cheng di tumori cerebrali (figshare, CC BY 4.0).

Si scarica UNA delle quattro parti pubbliche (766 slice T1w con contrasto) e la si
estrae in data/tumor/images/. Idempotente. I dati grezzi NON sono committati
(cartella data/ gitignorata).

Riferimento: Cheng J. et al., brain tumor dataset, figshare,
DOI 10.6084/m9.figshare.1512427 (CC BY 4.0).

Esegui:  uv run python scripts/download_tumor.py
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import requests

# Parte 1/4 del dataset pubblico (slice 1-766).
URL = "https://ndownloader.figshare.com/files/3381290"
DEST = Path(__file__).resolve().parents[1] / "data" / "tumor"
ZIP_PATH = DEST / "part1.zip"
IMAGES_DIR = DEST / "images"


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    print("Dataset: Cheng brain tumor (figshare) — licenza CC BY 4.0")

    if not (ZIP_PATH.exists() and ZIP_PATH.stat().st_size > 0):
        print("Scarico la parte 1/4 (~204 MB)...")
        with requests.get(URL, stream=True, timeout=600) as r:
            r.raise_for_status()
            with ZIP_PATH.open("wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
    else:
        print(f"[skip] zip già presente: {ZIP_PATH.name}")

    existing = list(IMAGES_DIR.glob("*.mat")) if IMAGES_DIR.exists() else []
    if not existing:
        IMAGES_DIR.mkdir(exist_ok=True)
        print("Estraggo i .mat...")
        with zipfile.ZipFile(ZIP_PATH) as zf:
            for name in zf.namelist():
                if name.endswith(".mat"):
                    data = zf.read(name)
                    (IMAGES_DIR / Path(name).name).write_bytes(data)
    n = len(list(IMAGES_DIR.glob("*.mat")))
    print(f"Pronte {n} slice in {IMAGES_DIR}")
    print("Citazione: Cheng et al., DOI 10.6084/m9.figshare.1512427 (CC BY 4.0).")


if __name__ == "__main__":
    main()
