"""Smoke test of the Streamlit app: the first render runs without exceptions.

Without a checkpoint the app stops with an informational message (no download,
no model loaded): that is the path tested here.
"""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app" / "main.py"


def test_app_initial_render():
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception
    assert any("medical image classification" in t.value for t in at.title)
