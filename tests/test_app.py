"""Smoke test for the Streamlit app (skipped when streamlit is not installed)."""

import shutil
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def app(tmp_path, monkeypatch):
    shutil.copy(ROOT / "data" / "sample_matches.csv", tmp_path / "matches.csv")
    (tmp_path / "ratings.csv").write_text("player,mine\nAndrea,9\nFabio,1\n")
    monkeypatch.setenv("QET_DATA", str(tmp_path / "matches.csv"))
    monkeypatch.setenv("QET_RATINGS", str(tmp_path / "ratings.csv"))
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()


def test_app_renders_all_tabs(app):
    assert not app.exception
    assert [t.label for t in app.tabs] == ["Ratings", "Balance", "Predict", "Evaluate", "Matches"]


def test_balance_shows_options(app):
    players = ["Marco", "Luca", "Stefano", "Paolo", "Andrea", "Giorgio",
               "Davide", "Matteo", "Simone", "Fabio", "Nicola", "Alberto"]
    app.multiselect(key="balance_known").set_value(players).run()
    assert not app.exception
    assert [s.value for s in app.subheader] == ["Option 1", "Option 2", "Option 3"]
