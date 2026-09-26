import pytest

from qet_football import Match, RatingModel, balance_teams, load_matches
from qet_football.data import parse_team, sheets_csv_url

SAMPLE = "data/sample_matches.csv"


def test_load_sample():
    matches = load_matches(SAMPLE)
    assert len(matches) == 6
    assert all(len(m.team_a) == 6 and len(m.team_b) == 6 for m in matches)


def test_parse_team_normalizes_names():
    assert parse_team(" marco ; LUCA;;") == ("Marco", "Luca")


def test_sheets_url_conversion():
    url = "https://docs.google.com/spreadsheets/d/abc_123/edit#gid=42"
    assert sheets_csv_url(url) == (
        "https://docs.google.com/spreadsheets/d/abc_123/export?format=csv&gid=42"
    )
    assert sheets_csv_url("data/x.csv") == "data/x.csv"


def test_strong_player_gets_positive_rating():
    # "Star" wins every game by 3 regardless of teammates.
    others = [f"P{i}" for i in range(11)]
    matches = []
    for k in range(10):
        rot = others[k % 11:] + others[: k % 11]
        matches.append(Match(("Star", *rot[:5]), tuple(rot[5:11]), 5, 2))
    model = RatingModel().fit(matches)
    assert model.rating("star") == max(model.ratings.values()) > 0


def test_prediction_is_symmetric_and_sums_to_one():
    model = RatingModel().fit(load_matches(SAMPLE))
    a = ["Marco", "Luca", "Stefano", "Paolo", "Andrea", "Giorgio"]
    b = ["Davide", "Matteo", "Simone", "Fabio", "Nicola", "Alberto"]
    p, q = model.predict(a, b), model.predict(b, a)
    assert p.p_win_a + p.p_draw + p.p_win_b == pytest.approx(1.0)
    assert p.goal_diff == pytest.approx(-q.goal_diff)
    assert p.p_win_a == pytest.approx(q.p_win_b)


def test_unknown_player_is_average():
    model = RatingModel().fit(load_matches(SAMPLE))
    assert model.rating("Newcomer") == 0.0


def test_balance_finds_best_split_and_respects_constraints():
    model = RatingModel()
    model.ratings = {f"P{i}": float(i) for i in range(12)}
    players = list(model.ratings)
    best = balance_teams(players, model, top=1)[0]
    assert best.goal_diff == 0.0
    assert len(best.team_a) == len(best.team_b) == 6

    splits = balance_teams(players, model, top=500, together=[["P10", "P11"]], apart=[["P0", "P1"]])
    for s in splits:
        assert ("P10" in s.team_a) == ("P11" in s.team_a)
        assert ("P0" in s.team_a) != ("P1" in s.team_a)


def test_balance_counts_all_splits():
    model = RatingModel()
    assert len(balance_teams([f"P{i}" for i in range(12)], model, top=10_000)) == 462
