"""Scarica PneumoniaMNIST (MedMNIST) in data/ — idempotente.

Licenza dataset: CC BY 4.0. Fonte: medmnist.com / Yang et al., Sci Data 2023
(DOI 10.1038/s41597-022-01721-8). Nessuna registrazione richiesta.
"""

from __future__ import annotations

from cxr_xai.data import CLASS_NAMES, load_split


def main() -> None:
    for split in ("train", "val", "test"):
        loader = load_split(split, batch_size=128)
        n = len(loader.dataset)
        print(f"{split:5s}: {n} immagini")
    print(f"Classi: {CLASS_NAMES}")
    print("Download completato (o già presente) in data/.")


if __name__ == "__main__":
    main()
