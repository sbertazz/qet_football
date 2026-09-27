"""Player rating model.

Each player has a rating r_p = goals they add to their team's goal difference
versus an average player. For a match, the expected goal difference is

    E[goals_a - goals_b] = sum(r_p for p in team_a) - sum(r_p for p in team_b)

Ratings are fitted with ridge regression, i.e. a Gaussian prior on each rating:
with few matches, players are pulled towards their prior. Guests are rated 0.

Manual scores (1-10, optional) set that prior: a player's starting point is
    beta * (score - mean score)
where beta, the goals one score point is worth, is fitted from the matches
too (itself pulled towards BETA_PRIOR). Players without a score start at 0,
i.e. average. The residual spread gives win/draw/loss probabilities.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .data import Match, is_guest, normalize_name

# Floor on goal-difference std dev: with little data the fitted residuals are
# too optimistic, and a 6v6 game is noisy.
MIN_SIGMA = 2.0

# Goals per manual score point before seeing any match, and how firmly beta is
# held there (higher = trust the prior more). Rating 9 vs 3 ~ 1 goal per game.
BETA_PRIOR = 0.15
BETA_PENALTY = 20.0


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
        self.scores: dict[str, float] = {}
        self.beta = 0.0
        self.sigma = MIN_SIGMA

    def fit(
        self, matches: Sequence[Match], scores: Mapping[str, float] | None = None
    ) -> RatingModel:
        """Fit on match results; `scores` are optional manual 1-10 ratings."""
        self.scores = {normalize_name(p): float(v) for p, v in (scores or {}).items()}
        players = sorted(
            {p for m in matches for p in (*m.team_a, *m.team_b) if not is_guest(p)}
            | set(self.scores)
        )
        index = {p: i for i, p in enumerate(players)}
        X = np.zeros((len(matches), len(players)))
        y = np.array([m.goal_diff for m in matches], dtype=float)
        for row, m in enumerate(matches):
            # Guests have no column: they count as an average (0) player.
            for p in m.team_a:
                if p in index:
                    X[row, index[p]] += 1
            for p in m.team_b:
                if p in index:
                    X[row, index[p]] -= 1

        # rating = beta * centred score + delta. Solve for [delta..., beta] jointly,
        # with delta pulled to 0 (strength alpha) and beta to BETA_PRIOR.
        mean_score = float(np.mean(list(self.scores.values()))) if self.scores else 0.0
        centred = np.array([self.scores.get(p, mean_score) - mean_score for p in players])
        n = len(players)
        A = np.column_stack([X, X @ centred])
        penalty = np.diag([self.alpha] * n + [BETA_PENALTY])
        prior = np.zeros(n + 1)
        if self.scores:
            prior[n] = BETA_PRIOR
        else:
            penalty[n, n] = 1.0  # beta has nothing to act on; keep the system solvable
        theta = np.linalg.solve(A.T @ A + penalty, A.T @ y + penalty @ prior)
        self.beta = float(theta[n]) if self.scores else 0.0
        w = theta[:n] + self.beta * centred
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

    def table(self) -> list[tuple[str, float, int, float | None]]:
        """(player, rating, games played, manual score or None), best first."""
        return sorted(
            ((p, r, self.games[p], self.scores.get(p)) for p, r in self.ratings.items()),
            key=lambda t: t[1],
            reverse=True,
        )
