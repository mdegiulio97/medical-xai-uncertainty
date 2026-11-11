"""Download PneumoniaMNIST (MedMNIST) into data/ (idempotent).

Dataset licence: CC BY 4.0. Source: medmnist.com / Yang et al., Sci Data 2023
(DOI 10.1038/s41597-022-01721-8). No registration required.
"""

from __future__ import annotations

from cxr_xai.data import CLASS_NAMES, load_split


def main() -> None:
    for split in ("train", "val", "test"):
        loader = load_split(split, batch_size=128)
        n = len(loader.dataset)
        print(f"{split:5s}: {n} images")
    print(f"Classes: {CLASS_NAMES}")
    print("Download complete (or already present) in data/.")


if __name__ == "__main__":
    main()
