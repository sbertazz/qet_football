"""Loading match results.

Expected columns (one row per match):
    date, team_a, team_b, goals_a, goals_b
where team_a / team_b are player names separated by ';'.

The source can be a local CSV or a Google Sheets link. For a sheet, share it
as "Anyone with the link can view"; edit links are converted to CSV exports.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

PLAYER_SEP = ";"
REQUIRED_COLUMNS = ["team_a", "team_b", "goals_a", "goals_b"]

_SHEETS_RE = re.compile(r"docs\.google\.com/spreadsheets/d/([\w-]+)")
_GID_RE = re.compile(r"[#&?]gid=(\d+)")


@dataclass(frozen=True)
class Match:
    team_a: tuple[str, ...]
    team_b: tuple[str, ...]
    goals_a: int
    goals_b: int
    date: str | None = None

    @property
    def goal_diff(self) -> int:
        return self.goals_a - self.goals_b


def normalize_name(name: str) -> str:
    """Trim and title-case so 'marco ' and 'Marco' are the same player."""
    return " ".join(name.split()).title()


def parse_team(cell: str) -> tuple[str, ...]:
    players = [normalize_name(p) for p in str(cell).split(PLAYER_SEP)]
    return tuple(p for p in players if p)


def sheets_csv_url(url: str) -> str:
    """Turn a Google Sheets link into its CSV export URL; other sources pass through."""
    m = _SHEETS_RE.search(url)
    if not m or "export?format=csv" in url:
        return url
    gid = _GID_RE.search(url)
    export = f"https://docs.google.com/spreadsheets/d/{m.group(1)}/export?format=csv"
    return export + (f"&gid={gid.group(1)}" if gid else "")


def load_matches(source: str) -> list[Match]:
    df = pd.read_csv(sheets_csv_url(source))
    df.columns = [c.strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns {missing}; found {list(df.columns)}")
    df = df.dropna(subset=REQUIRED_COLUMNS)

    matches = []
    for row in df.itertuples(index=False):
        team_a, team_b = parse_team(row.team_a), parse_team(row.team_b)
        overlap = set(team_a) & set(team_b)
        if overlap:
            raise ValueError(f"players on both teams in match {row}: {sorted(overlap)}")
        matches.append(
            Match(
                team_a=team_a,
                team_b=team_b,
                goals_a=int(row.goals_a),
                goals_b=int(row.goals_b),
                date=str(row.date) if "date" in df.columns else None,
            )
        )
    return matches
