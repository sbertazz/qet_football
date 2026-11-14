"""Player rating model.

Each player has a rating r_p = goals they add to their team's goal difference
versus an average player. For a match, the expected goal difference is

    E[goals_a - goals_b] = sum(r_p for p in team_a) - sum(r_p for p in team_b)

Ratings are fitted with ridge regression, i.e. a Gaussian prior on each rating:
with few matches, players are pulled towards their prior. Guests are rated 0.

Manual scores (1-10, optional) set that prior: a player's starting point is
    beta * (score - RATING_MIDPOINT)
where beta is the goals one score point is worth: fixed (DEFAULT_BETA) or, with
beta=None, fitted from the matches too (pulled towards BETA_PRIOR). Rated
players are held at their starting point `hold` times more firmly than
unrated ones, who start at 0, i.e. average (a 5.5 on the 1-10 scale). The residual spread gives
win/draw/loss probabilities.
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

# Each match's goal difference is clipped to +/- this many goals when fitting,
# so a single blowout (often partly luck) can't dominate the ratings.
DEFAULT_CAP = 4

# Middle of the 1-10 manual scale: an unrated player counts as a 5.5, whoever
# else is rated. (Centring on the mean of the given scores made an 8 count as
# below average when only strong players were rated.)
RATING_MIDPOINT = 5.5

# Goals per manual score point (10 vs 1 ~ 1.8 goals per game), and how many
# times more firmly rated players are held at their starting point than
# unrated ones. With few matches, learning beta from the data made the manual
# ratings collapse to ~0 although trusting them predicted unseen matches better.
DEFAULT_BETA = 0.2
DEFAULT_HOLD = 20.0

# Only when beta is learned (beta=None): its starting value and how firmly it
# is held there.
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
    def __init__(
        self,
        alpha: float = 3.0,
        cap: float | None = DEFAULT_CAP,
        beta: float | None = DEFAULT_BETA,
        hold: float = DEFAULT_HOLD,
    ):
        self.alpha = alpha
        self.cap = cap or None  # None or 0 = no cap
        self.fixed_beta = beta  # None = learn beta from the matches
        self.hold = hold
        self.ratings: dict[str, float] = {}
        self.games: dict[str, int] = {}
        self.won: dict[str, float] = {}  # win = 1, draw = 0.5, loss = 0
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
        if self.cap is not None:
            y = np.clip(y, -self.cap, self.cap)
        for row, m in enumerate(matches):
            # Guests have no column: they count as an average (0) player.
            for p in m.team_a:
                if p in index:
                    X[row, index[p]] += 1
            for p in m.team_b:
                if p in index:
                    X[row, index[p]] -= 1

        # rating = beta * centred score + delta, with delta pulled to 0: strength
        # alpha, times `hold` for rated players.
        centred = np.array(
            [self.scores[p] - RATING_MIDPOINT if p in self.scores else 0.0 for p in players]
        )
        n = len(players)
        delta_penalty = np.array(
            [self.alpha * (self.hold if p in self.scores else 1.0) for p in players]
        )
        if not self.scores:
            self.beta = 0.0
        elif self.fixed_beta is not None:
            self.beta = float(self.fixed_beta)
        if not self.scores or self.fixed_beta is not None:
            target = y - X @ (self.beta * centred)
            delta = np.linalg.solve(X.T @ X + np.diag(delta_penalty), X.T @ target)
        else:
            # Learned beta: solve for [delta..., beta] jointly, beta pulled to BETA_PRIOR.
            A = np.column_stack([X, X @ centred])
            penalty = np.diag([*delta_penalty, BETA_PENALTY])
            prior = np.zeros(n + 1)
            prior[n] = BETA_PRIOR
            theta = np.linalg.solve(A.T @ A + penalty, A.T @ y + penalty @ prior)
            delta, self.beta = theta[:n], float(theta[n])
        w = delta + self.beta * centred
        self.ratings = dict(zip(players, w.tolist()))
        self.games = {p: int(np.count_nonzero(X[:, i])) for p, i in index.items()}
        self.won = dict.fromkeys(players, 0.0)
        for m in matches:
            points_a = 1.0 if m.goal_diff > 0 else 0.5 if m.goal_diff == 0 else 0.0
            for team, points in ((m.team_a, points_a), (m.team_b, 1.0 - points_a)):
                for p in team:
                    if p in self.won:
                        self.won[p] += points

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
