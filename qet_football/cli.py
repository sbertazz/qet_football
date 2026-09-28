"""Command line interface.

    python qet.py ratings
    python qet.py predict --a "Marco,Luca,..." --b "Davide,Matteo,..."
    python qet.py balance "Marco,Luca,Davide,..." [--together "Marco,Luca"] [--apart "A,B"]
    python qet.py evaluate

The data source is --data, else $QET_DATA, else data/matches.csv.
Manual 1-10 ratings are read from --ratings, else $QET_RATINGS, else
data/player_ratings.csv if it exists. That file can hold several versions
(one column each); predict/balance use --version, else the first column.
"""

from __future__ import annotations

import argparse
import os

from .balance import balance_teams
from .data import Match, load_matches, load_score_versions, normalize_name
from .evaluate import leave_one_out, summarize
from .model import RatingModel

DEFAULT_DATA = "data/matches.csv"
DEFAULT_RATINGS = "data/player_ratings.csv"
NO_RATINGS = "Unadjusted"  # match results only, no manual ratings


def _names(text: str) -> list[str]:
    return [p.strip() for p in text.replace(";", ",").split(",") if p.strip()]


def _load_matches(args: argparse.Namespace) -> list[Match]:
    source = args.data or os.environ.get("QET_DATA") or DEFAULT_DATA
    matches = load_matches(source)
    print(f"{len(matches)} matches from {source}")
    return matches


def _load_versions(args: argparse.Namespace) -> dict[str, dict[str, float]]:
    """NO_RATINGS first, then all manual rating versions."""
    source = args.ratings or os.environ.get("QET_RATINGS")
    if not source and os.path.exists(DEFAULT_RATINGS):
        source = DEFAULT_RATINGS
    versions = load_score_versions(source) if source else {}
    if versions:
        print(f"Rating versions from {source}: {', '.join(versions)}")
    return {NO_RATINGS: {}, **versions}


def _load_model(args: argparse.Namespace) -> RatingModel:
    matches, versions = _load_matches(args), _load_versions(args)
    if args.version:
        name = next((v for v in versions if v.lower() == args.version.lower()), None)
        if name is None:
            raise SystemExit(f"unknown version {args.version!r}; choose from {', '.join(versions)}")
    else:  # first manual version if there is one
        name = next((v for v in versions if v != NO_RATINGS), NO_RATINGS)
    model = RatingModel(alpha=args.alpha).fit(matches, versions[name])
    detail = f" (1 point = {model.beta:.2f} goals)" if versions[name] else ""
    print(f"Using ratings: {name}{detail}\n")
    return model


def _col(name: str) -> int:
    return max(len(name), 6) + 2


def cmd_ratings(args: argparse.Namespace) -> None:
    matches, versions = _load_matches(args), _load_versions(args)
    models = {v: RatingModel(alpha=args.alpha).fit(matches, s) for v, s in versions.items()}
    for v, model in models.items():
        if versions[v]:
            print(f"  {v}: {len(versions[v])} players rated, 1 point = {model.beta:.2f} goals")
    print()
    # Rated players who never played appear only in the manual versions.
    players = {p for m in models.values() for p in m.ratings}
    unadjusted = models[NO_RATINGS]
    header = "".join(f"{v:>{_col(v)}}" for v in models)
    print(f"{'Player':<20}{header}{'Games':>7}{'Won':>6}")
    for p in sorted(players, key=lambda p: (-unadjusted.rating(p), p)):
        cells = "".join(f"{m.rating(p):>+{_col(v)}.2f}" for v, m in models.items())
        games, won = unadjusted.games.get(p, 0), unadjusted.won.get(p, 0.0)
        print(f"{p:<20}{cells}{games:>7}{won:>6.1f}")


def cmd_predict(args: argparse.Namespace) -> None:
    model = _load_model(args)
    a, b = _names(args.a), _names(args.b)
    p = model.predict(a, b)
    for label, team in (("A", a), ("B", b)):
        unknown = [x for x in team if normalize_name(x) not in model.ratings]
        note = f"  (no history: {', '.join(unknown)})" if unknown else ""
        print(f"Team {label}: {', '.join(team)}{note}")
    print(f"\nExpected goal difference (A - B): {p.goal_diff:+.1f}  (+/- {p.sigma:.1f})")
    print(f"A wins {p.p_win_a:.0%}   draw {p.p_draw:.0%}   B wins {p.p_win_b:.0%}")


def cmd_balance(args: argparse.Namespace) -> None:
    model = _load_model(args)
    splits = balance_teams(
        _names(args.players),
        model,
        top=args.top,
        together=[_names(g) for g in args.together],
        apart=[_names(g) for g in args.apart],
    )
    if not splits:
        print("No split satisfies the constraints.")
        return
    for i, s in enumerate(splits, 1):
        print(f"Option {i}  (expected diff {s.goal_diff:+.2f})")
        print(f"  A: {', '.join(s.team_a)}")
        print(f"  B: {', '.join(s.team_b)}\n")


def cmd_evaluate(args: argparse.Namespace) -> None:
    matches, versions = _load_matches(args), _load_versions(args)
    results = leave_one_out(matches, versions, alpha=args.alpha)
    print("Each match predicted from all the others (goal difference, left - right):\n")
    header = f"{'Match':<12}{'Score':>7}{'Actual':>8}" + "".join(f"{v:>{_col(v)}}" for v in results)
    print(header)
    for i, m in enumerate(matches):
        cells = "".join(f"{results[v][i].goal_diff:>+{_col(v)}.1f}" for v in results)
        score = f"{m.goals_a}-{m.goals_b}"
        print(f"{(m.date or f'#{i + 1}'):<12}{score:>7}{m.goal_diff:>+8d}{cells}")

    summaries = {v: summarize(matches, preds) for v, preds in results.items()}
    rows = [
        ("Avg error (goals)", lambda s: s.mean_error, min, lambda x: f"{x:.2f}"),
        ("Outcome right", lambda s: s.correct, max, lambda x: f"{x}/{len(matches)}"),
        ("Avg P(actual)", lambda s: s.mean_p_actual, max, lambda x: f"{x:.0%}"),
    ]
    print("-" * len(header))
    for label, get, best_fn, fmt in rows:
        best = best_fn(get(s) for s in summaries.values())
        cells = "".join(
            f"{fmt(get(s)) + ('*' if get(s) == best else ''):>{_col(v)}}"
            for v, s in summaries.items()
        )
        print(f"{label:<27}{cells}")
    print("\n* = best. Lower error is better; higher outcome-right and P(actual) are better.")
    if len(matches) < 10:
        print(f"Only {len(matches)} matches: differences between versions may just be luck.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python qet.py", description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", help="CSV path or Google Sheets link")
    parser.add_argument("--ratings", help="CSV of manual 1-10 ratings (player + one column per version)")
    parser.add_argument("--version", help="which ratings column predict/balance use (default: first)")
    parser.add_argument("--alpha", type=float, default=3.0,
                        help="shrinkage towards average; higher = more cautious ratings")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("ratings", help="show player ratings").set_defaults(func=cmd_ratings)

    sub.add_parser("evaluate", help="compare rating versions on past matches"
                   ).set_defaults(func=cmd_evaluate)

    p = sub.add_parser("predict", help="predict a match between two teams")
    p.add_argument("--a", required=True, help="team A players, comma separated")
    p.add_argument("--b", required=True, help="team B players, comma separated")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("balance", help="split players into two balanced teams")
    p.add_argument("players", help="all players, comma separated")
    p.add_argument("--top", type=int, default=3, help="how many options to show")
    p.add_argument("--together", action="append", default=[],
                   help="players who must play together (repeatable)")
    p.add_argument("--apart", action="append", default=[],
                   help="players who must be on different teams (repeatable)")
    p.set_defaults(func=cmd_balance)

    args = parser.parse_args(argv)
    args.func(args)
    return 0
