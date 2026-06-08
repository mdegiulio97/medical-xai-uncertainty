"""Smoke test dell'app Streamlit: il primo render gira senza eccezioni.

Senza checkpoint l'app si ferma con un messaggio informativo (nessun download,
nessun modello caricato): è il percorso che verifichiamo qui.
"""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app" / "main.py"


def test_app_initial_render():
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception
    assert any("Chest X-ray" in t.value for t in at.title)
