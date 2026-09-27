"""Search manual ratings that minimise the leave-one-out error.

This deliberately optimises against the leave-one-out test, so the result
overfits: the searched ratings explain past matches very well but are not
expected to predict new ones better than honest ratings. Kept for exploring.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence

from .data import Match
from .evaluate import leave_one_out, summarize

START_VALUE = 5


def loo_error(matches: Sequence[Match], scores: Mapping[str, float], alpha: float = 3.0) -> float:
    return summarize(matches, leave_one_out(matches, {"v": scores}, alpha)["v"]).mean_error


def search_ratings(
    matches: Sequence[Match],
    start: Mapping[str, float],
    players: Sequence[str],
    steps: int = 1500,
    seed: int = 0,
    alpha: float = 3.0,
) -> tuple[dict[str, float], float, float]:
    """Hill-climb 1-10 ratings for `players`, starting from `start`.

    Each step sets one random player to a random rating and keeps the change
    if the leave-one-out error does not get worse. Players missing from
    `start` begin at START_VALUE. Returns (ratings, error before, error after).
    """
    rng = random.Random(seed)
    current = {p: float(start.get(p, START_VALUE)) for p in players}
    before = best = loo_error(matches, current, alpha)
    for _ in range(steps):
        candidate = dict(current)
        candidate[rng.choice(players)] = float(rng.randint(1, 10))
        error = loo_error(matches, candidate, alpha)
        if error <= best:
            current, best = candidate, error
    return current, before, best
