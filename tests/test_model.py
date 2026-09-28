import pytest

from qet_football import Match, RatingModel, balance_teams, load_matches
from qet_football.data import load_score_versions
from qet_football.evaluate import leave_one_out, summarize
from qet_football.data import normalize_name, parse_team, sheets_csv_url

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


BLOCKS = "tests/data/blocks.csv"


def test_load_block_format():
    matches = load_matches(BLOCKS)
    assert len(matches) == 2
    first, second = matches
    assert first.date == "2026-09-08" and second.date == "2026-09-15"
    assert first.team_a == ("Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot")
    assert first.team_b[4] == "Kilo"  # no numbering or hidden characters left
    assert (first.goals_a, first.goals_b) == (12, 10)
    assert second.team_a[1] == "Bravo+1"
    assert second.team_b[2] == "India"  # case normalized


def test_normalize_strips_numbering_and_invisible_chars():
    assert normalize_name("3. ⁠hasnain ") == "Hasnain"


def test_guests_are_not_rated():
    model = RatingModel().fit(load_matches(BLOCKS))
    assert "Bravo+1" not in model.ratings
    assert model.rating("Bravo+1") == 0.0
    assert "Bravo" in model.ratings


def test_block_without_score_is_an_error(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("Whites:,Bibs:,01/09/26\n1. A,1. B,\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no score"):
        load_matches(str(bad))


def test_load_scores_accepts_player_or_name_header(tmp_path):
    for header in ("player", "Name"):
        f = tmp_path / f"{header}.csv"
        f.write_text(f"{header},rating\n \u2060alpha,7\nBravo,3\nCharlie,\n", encoding="utf-8")
        assert load_score_versions(str(f)) == {"rating": {"Alpha": 7.0, "Bravo": 3.0}}


def test_load_several_versions(tmp_path):
    f = tmp_path / "r.csv"
    f.write_text("player,ratings1,ratings2\nAlpha,7,8\nBravo,,4\n", encoding="utf-8")
    versions = load_score_versions(str(f))
    assert list(versions) == ["ratings1", "ratings2"]
    assert versions["ratings1"] == {"Alpha": 7.0}  # empty cell = not rated
    assert versions["ratings2"] == {"Alpha": 8.0, "Bravo": 4.0}


def test_load_scores_rejects_out_of_range(tmp_path):
    f = tmp_path / "r.csv"
    f.write_text("player,rating\nAlpha,11\n", encoding="utf-8")
    with pytest.raises(ValueError, match="1-10"):
        load_score_versions(str(f))


def test_leave_one_out_and_summary():
    matches = load_matches(SAMPLE)
    versions = {"mine": {"Andrea": 9, "Fabio": 2}, "Unadjusted": {}}
    results = leave_one_out(matches, versions)
    assert set(results) == {"mine", "Unadjusted"}
    assert all(len(preds) == len(matches) for preds in results.values())

    # Prediction for match 0 must not have seen match 0.
    expected = RatingModel().fit(matches[1:], {}).predict(matches[0].team_a, matches[0].team_b)
    assert results["Unadjusted"][0].goal_diff == pytest.approx(expected.goal_diff)

    s = summarize(matches, results["Unadjusted"])
    assert s.mean_error >= 0
    assert 0 <= s.correct <= len(matches)
    assert 0 < s.mean_p_actual < 1


def test_leave_one_out_needs_two_matches():
    with pytest.raises(ValueError):
        leave_one_out(load_matches(SAMPLE)[:1], {"Unadjusted": {}})


def test_no_scores_matches_plain_model():
    matches = load_matches(SAMPLE)
    plain = RatingModel().fit(matches)
    empty = RatingModel().fit(matches, {})
    assert plain.ratings == pytest.approx(empty.ratings)
    assert empty.beta == 0.0


def test_scores_set_the_starting_point():
    # With no matches, ratings are exactly beta * (score - mean score).
    model = RatingModel().fit([], {"High": 9, "Low": 3, "Mid": 6})
    assert model.rating("High") > model.rating("Mid") > model.rating("Low")
    assert model.rating("Mid") == pytest.approx(0.0)
    assert model.rating("Nobody") == 0.0


def test_scores_that_agree_with_results_increase_beta():
    # The team with more highly rated players always wins by 3.
    good, bad = [f"G{i}" for i in range(6)], [f"B{i}" for i in range(6)]
    matches = []
    for k in range(8):
        g = good[k % 6:] + good[: k % 6]
        matches.append(Match((*g[:4], *bad[:2]), (*g[4:], *bad[2:]), 6, 3))
    scores = {**{p: 8 for p in good}, **{p: 3 for p in bad}}
    model = RatingModel().fit(matches, scores)
    assert model.beta > 0.15


def test_load_scores_skips_comment_lines(tmp_path):
    f = tmp_path / "r.csv"
    f.write_text("# my notes\nplayer,rating\nAlpha,7\n", encoding="utf-8")
    assert load_score_versions(str(f)) == {"rating": {"Alpha": 7.0}}


def test_ratings_command_puts_unadjusted_first_and_sorts_by_it(tmp_path, monkeypatch, capsys):
    import shutil

    from qet_football.cli import main

    (tmp_path / "data").mkdir()
    shutil.copy(SAMPLE, tmp_path / "data" / "matches.csv")
    (tmp_path / "data" / "player_ratings.csv").write_text("player,mine\nFabio,10\nNewbie,9\n")
    monkeypatch.chdir(tmp_path)

    main(["ratings"])
    lines = capsys.readouterr().out.splitlines()
    header = next(line for line in lines if line.startswith("Player"))
    assert header.split() == ["Player", "Unadjusted", "mine", "Games", "Won"]
    rows = [line.split() for line in lines[lines.index(header) + 1:]]
    unadjusted = [float(r[1]) for r in rows]
    assert unadjusted == sorted(unadjusted, reverse=True)
    assert "Newbie" in [r[0] for r in rows]  # rated but never played

    main(["predict", "--a", "Marco", "--b", "Luca"])
    assert "Using ratings: mine" in capsys.readouterr().out  # default stays the first manual version
    main(["--version", "unadjusted", "predict", "--a", "Marco", "--b", "Luca"])
    assert "Using ratings: Unadjusted" in capsys.readouterr().out


def test_won_counts_wins_and_half_draws():
    model = RatingModel().fit(load_matches(BLOCKS))
    # 08/09: left 12-10 (Alpha's team wins); 15/09: right 9-7 (Alpha's team wins again)
    assert model.won["Alpha"] == 2.0
    assert model.won["Golf"] == 0.0  # lost both
    assert "Bravo+1" not in model.won

    draw = [Match(("A", "B"), ("C", "D"), 3, 3), Match(("A", "C"), ("B", "D"), 5, 2)]
    won = RatingModel().fit(draw).won
    assert won == {"A": 1.5, "B": 0.5, "C": 1.5, "D": 0.5}
