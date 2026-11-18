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
        "Ratings", "Matches", "Manual ratings", "Balance", "Predict", "Evaluate"
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


def test_all_matches_included_by_default(app):
    caption = " ".join(c.value for c in app.sidebar.caption)
    assert caption.startswith("6 matches")
    assert not any("excluded" in i.value for i in app.info)


def test_contribution_charts_render(app):
    # Matches tab always shows the chart for the latest match.
    assert len(app.get("vega_lite_chart")) == 1
    players = ["Marco", "Luca", "Stefano", "Paolo", "Andrea", "Giorgio",
               "Davide", "Matteo", "Simone", "Fabio", "Nicola", "Alberto"]
    app.multiselect(key="pred_a_known").set_value(players[:6])
    app.multiselect(key="pred_b_known").set_value(players[6:]).run()
    assert not app.exception
    assert len(app.get("vega_lite_chart")) == 2
    assert any("Expected:" in m.value for m in app.markdown)
    # Text version of the chart, one line per team with net total and players best first.
    team_lines = [m.value for m in app.markdown if m.value.startswith("**Team A · net")]
    assert team_lines and all(p in team_lines[-1] for p in players[:6])


def test_ratings_change_since_last_match_day(app):
    table = app.dataframe[0].value
    assert list(table.columns[:5]) == ["Player", "Unadjusted", "Unadjusted Δ", "mine", "mine Δ"]
    # Sample data: the last match day (2026-02-12) is one match; everyone played before it.
    changes = list(table["Unadjusted Δ"])
    assert "new" not in changes
    assert any(c.startswith(("+", "-")) for c in changes)
    assert any("change from the last match day (2026-02-12" in c.value for c in app.caption)


def _fresh_app():
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)


def test_password_gate(tmp_path, monkeypatch):
    shutil.copy(ROOT / "data" / "sample_matches.csv", tmp_path / "matches.csv")
    monkeypatch.setenv("QET_DATA", str(tmp_path / "matches.csv"))
    monkeypatch.setenv("QET_PASSWORD", "secret")
    at = _fresh_app().run()
    assert not at.tabs  # nothing but the password box
    at.text_input(key="password").set_value("wrong").run()
    assert not at.tabs and at.error
    at.text_input(key="password").set_value("secret").run()
    assert not at.exception
    assert [t.label for t in at.tabs][0] == "Ratings"


def test_viewer_mode_hides_files_and_saving(tmp_path, monkeypatch):
    shutil.copy(ROOT / "data" / "sample_matches.csv", tmp_path / "matches.csv")
    monkeypatch.setenv("QET_DATA", str(tmp_path / "matches.csv"))
    monkeypatch.setenv("QET_VIEWER", "true")
    monkeypatch.delenv("QET_RATINGS", raising=False)
    at = _fresh_app().run()
    assert not at.exception
    labels = [t.label for t in at.text_input]
    assert not any("Match data" in label or "ratings file" in label for label in labels)
    assert not any(b.label == "Save to file" for b in at.button)
    assert at.sidebar.selectbox[0].options == ["Unadjusted"]  # no manual ratings to start


def test_unreachable_sheet_shows_sharing_hint(tmp_path, monkeypatch):
    monkeypatch.setenv("QET_DATA", "http://127.0.0.1:9/not-a-sheet")
    at = _fresh_app().run()
    assert at.error and "shared" in at.error[0].value
