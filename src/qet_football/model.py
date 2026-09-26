"""Player rating model.

Each player has a rating r_p = goals they add to their team's goal difference
versus an average player. For a match, the expected goal difference is

    E[goals_a - goals_b] = sum(r_p for p in team_a) - sum(r_p for p in team_b)

Ratings are fitted with ridge regression, i.e. a Gaussian prior centred on 0:
with few matches, players are pulled towards "average", and unknown players
are rated exactly 0. The residual spread gives win/draw/loss probabilities.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from .data import Match, normalize_name

# Floor on goal-difference std dev: with little data the fitted residuals are
# too optimistic, and a 6v6 game is noisy.
MIN_SIGMA = 2.0


@dataclass(frozen=True)
class Prediction:
    goal_diff: float  # expected goals_a - goals_b
    sigma: float
    p_win_a: float
    p_draw: float
    p_win_b: float


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


class RatingModel:
    def __init__(self, alpha: float = 3.0):
        self.alpha = alpha
        self.ratings: dict[str, float] = {}
        self.games: dict[str, int] = {}
        self.sigma = MIN_SIGMA

    def fit(self, matches: Sequence[Match]) -> RatingModel:
        players = sorted({p for m in matches for p in (*m.team_a, *m.team_b)})
        index = {p: i for i, p in enumerate(players)}
        X = np.zeros((len(matches), len(players)))
        y = np.array([m.goal_diff for m in matches], dtype=float)
        for row, m in enumerate(matches):
            for p in m.team_a:
                X[row, index[p]] += 1
            for p in m.team_b:
                X[row, index[p]] -= 1

        w = np.linalg.solve(X.T @ X + self.alpha * np.eye(len(players)), X.T @ y)
        self.ratings = dict(zip(players, w.tolist()))
        self.games = {p: int(np.count_nonzero(X[:, i])) for p, i in index.items()}

        dof = max(len(matches) - 1, 1)
        rms = math.sqrt(float(np.sum((y - X @ w) ** 2)) / dof)
        self.sigma = max(rms, MIN_SIGMA)
        return self

    def rating(self, player: str) -> float:
        return self.ratings.get(normalize_name(player), 0.0)

    def strength(self, team: Iterable[str]) -> float:
        return sum(self.rating(p) for p in team)

    def predict(self, team_a: Iterable[str], team_b: Iterable[str]) -> Prediction:
        mu = self.strength(team_a) - self.strength(team_b)
        # Goal difference is an integer: a draw is |diff| < 0.5 on the continuous scale.
        p_win_a = 1.0 - _norm_cdf((0.5 - mu) / self.sigma)
        p_win_b = _norm_cdf((-0.5 - mu) / self.sigma)
        return Prediction(mu, self.sigma, p_win_a, 1.0 - p_win_a - p_win_b, p_win_b)

    def table(self) -> list[tuple[str, float, int]]:
        """(player, rating, games played), best first."""
        return sorted(
            ((p, r, self.games[p]) for p, r in self.ratings.items()),
            key=lambda t: t[1],
            reverse=True,
        )
