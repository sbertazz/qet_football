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
    (tmp_path / "ratings.csv").write_text("player,mine,empty\nAndrea,9,\nFabio,1,\n")
    monkeypatch.setenv("QET_DATA", str(tmp_path / "matches.csv"))
    monkeypatch.setenv("QET_RATINGS", str(tmp_path / "ratings.csv"))
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()


def test_app_renders_all_tabs(app):
    assert not app.exception
    assert [t.label for t in app.tabs] == [
        "Ratings", "Balance", "Predict", "Evaluate", "Manual ratings", "Matches"
    ]


def test_balance_shows_options(app):
    players = ["Marco", "Luca", "Stefano", "Paolo", "Andrea", "Giorgio",
               "Davide", "Matteo", "Simone", "Fabio", "Nicola", "Alberto"]
    app.multiselect(key="balance_known").set_value(players).run()
    assert not app.exception
    assert [s.value for s in app.subheader] == ["Option 1", "Option 2", "Option 3"]


def test_manual_ratings_loaded_and_empty_version_ignored(app):
    version_picker = app.sidebar.selectbox[0]
    assert version_picker.options == ["Unadjusted", "mine"]
    assert version_picker.value == "mine"
    assert app.text_input(key="vname0").value == "mine"


def test_rename_and_save_ratings(app, tmp_path):
    app.text_input(key="vname0").set_value("my view").run()
    assert app.sidebar.selectbox[0].options == ["Unadjusted", "my view"]
    next(b for b in app.button if b.label == "Save to file").click().run()
    assert not app.exception
    saved = (tmp_path / "ratings.csv").read_text().splitlines()
    assert saved == ["player,my view", "Andrea,9", "Fabio,1"]
