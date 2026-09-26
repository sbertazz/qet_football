"""Split a list of players into two balanced teams."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

from .data import normalize_name
from .model import RatingModel


@dataclass(frozen=True)
class Split:
    team_a: tuple[str, ...]
    team_b: tuple[str, ...]
    goal_diff: float  # predicted goals_a - goals_b


def balance_teams(
    players: Sequence[str],
    model: RatingModel,
    top: int = 5,
    together: Sequence[Sequence[str]] = (),
    apart: Sequence[Sequence[str]] = (),
) -> list[Split]:
    """Return the `top` most balanced splits, best first.

    Every split is checked exhaustively (462 for 12 players). `together` and
    `apart` are groups of players that must (not) end up on the same team.
    """
    names = [normalize_name(p) for p in players]
    if len(set(names)) != len(names):
        raise ValueError("duplicate players in list")
    if len(names) % 2:
        raise ValueError(f"need an even number of players, got {len(names)}")

    together_sets = [{normalize_name(p) for p in g} for g in together]
    apart_pairs = [
        (normalize_name(a), normalize_name(b))
        for g in apart
        for a, b in combinations(g, 2)
    ]

    first, rest = names[0], names[1:]
    size = len(names) // 2
    splits = []
    # Fixing the first player on team A avoids counting each split twice.
    for others in combinations(rest, size - 1):
        team_a = {first, *others}
        if any(0 < len(g & team_a) < len(g) for g in together_sets):
            continue
        if any((a in team_a) == (b in team_a) for a, b in apart_pairs):
            continue
        a = tuple(p for p in names if p in team_a)
        b = tuple(p for p in names if p not in team_a)
        splits.append(Split(a, b, model.strength(a) - model.strength(b)))

    splits.sort(key=lambda s: abs(s.goal_diff))
    return splits[:top]
