"""Command line interface.

    qet ratings
    qet predict --a "Marco,Luca,..." --b "Davide,Matteo,..."
    qet balance "Marco,Luca,Davide,..." [--together "Marco,Luca"] [--apart "A,B"]

The data source is --data, else $QET_DATA, else data/matches.csv.
"""

from __future__ import annotations

import argparse
import os

from .balance import balance_teams
from .data import load_matches, normalize_name
from .model import RatingModel

DEFAULT_DATA = "data/matches.csv"


def _names(text: str) -> list[str]:
    return [p.strip() for p in text.replace(";", ",").split(",") if p.strip()]


def _load_model(args: argparse.Namespace) -> RatingModel:
    source = args.data or os.environ.get("QET_DATA") or DEFAULT_DATA
    matches = load_matches(source)
    print(f"Fitted on {len(matches)} matches from {source}\n")
    return RatingModel(alpha=args.alpha).fit(matches)


def cmd_ratings(args: argparse.Namespace) -> None:
    model = _load_model(args)
    print(f"{'Player':<20}{'Rating':>8}{'Games':>7}")
    for player, rating, games in model.table():
        print(f"{player:<20}{rating:>+8.2f}{games:>7}")


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="qet", description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", help="CSV path or Google Sheets link")
    parser.add_argument("--alpha", type=float, default=3.0,
                        help="shrinkage towards average; higher = more cautious ratings")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("ratings", help="show player ratings").set_defaults(func=cmd_ratings)

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
