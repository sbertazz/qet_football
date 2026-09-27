"""Compare rating versions by leave-one-out: predict each match from all the others."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .data import Match
from .model import Prediction, RatingModel


@dataclass(frozen=True)
class Summary:
    mean_error: float  # average |predicted - actual goal difference|
    correct: int  # matches where the most likely outcome (A / draw / B) happened
    mean_p_actual: float  # average probability given to the actual outcome


def outcome(goal_diff: float) -> int:
    """1 = team A wins, 0 = draw, -1 = team B wins."""
    return (goal_diff > 0) - (goal_diff < 0)


def p_actual(p: Prediction, goal_diff: int) -> float:
    return {1: p.p_win_a, 0: p.p_draw, -1: p.p_win_b}[outcome(goal_diff)]


def most_likely(p: Prediction) -> int:
    return max(((p.p_win_a, 1), (p.p_draw, 0), (p.p_win_b, -1)))[1]


def leave_one_out(
    matches: Sequence[Match],
    versions: Mapping[str, Mapping[str, float]],
    alpha: float = 3.0,
) -> dict[str, list[Prediction]]:
    """{version: [prediction for match i, fitted without match i]}."""
    if len(matches) < 2:
        raise ValueError("need at least 2 matches to evaluate")
    results: dict[str, list[Prediction]] = {v: [] for v in versions}
    for i, m in enumerate(matches):
        rest = [*matches[:i], *matches[i + 1:]]
        for version, scores in versions.items():
            model = RatingModel(alpha=alpha).fit(rest, scores)
            results[version].append(model.predict(m.team_a, m.team_b))
    return results


def summarize(matches: Sequence[Match], predictions: Sequence[Prediction]) -> Summary:
    n = len(matches)
    return Summary(
        mean_error=sum(abs(p.goal_diff - m.goal_diff) for m, p in zip(matches, predictions)) / n,
        correct=sum(most_likely(p) == outcome(m.goal_diff) for m, p in zip(matches, predictions)),
        mean_p_actual=sum(p_actual(p, m.goal_diff) for m, p in zip(matches, predictions)) / n,
    )
