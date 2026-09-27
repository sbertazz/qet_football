"""Loading match results.

Two layouts are accepted and detected automatically.

Table: one row per match, with columns
    date, team_a, team_b, goals_a, goals_b
where team_a / team_b are player names separated by ';'.

Blocks (the layout of the group's sheet): per match, a header row with the two
team labels and the date, numbered player rows, then the score row:
    Whites:,Bibs:,08/09/26
    1. Hamza,1. Med,
    ...
    12,12,

A name containing '+' (e.g. 'Ayman+1') is a guest: they fill a spot but are
treated as an average player and never get a rating.

The source can be a local CSV or a Google Sheets link. For a sheet, share it
as "Anyone with the link can view"; edit links are converted to CSV exports.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

PLAYER_SEP = ";"
REQUIRED_COLUMNS = ["team_a", "team_b", "goals_a", "goals_b"]

_SHEETS_RE = re.compile(r"docs\.google\.com/spreadsheets/d/([\w-]+)")
_GID_RE = re.compile(r"[#&?]gid=(\d+)")
# Zero-width / invisible characters that sneak in when names are pasted.
_INVISIBLE_RE = re.compile("[\u00ad\u200b-\u200f\u2060-\u2064\ufeff]")
_NUMBERED_RE = re.compile(r"^\s*\d+\s*[.)]\s*")
_DATE_FORMATS = ("%d/%m/%y", "%d/%m/%Y", "%Y-%m-%d")


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
    """Clean up so ' marco', 'Marco' and '2. \u2060Marco' are the same player."""
    name = _NUMBERED_RE.sub("", _INVISIBLE_RE.sub("", name))
    return " ".join(name.split()).title()


def is_guest(name: str) -> bool:
    return "+" in name


def parse_date(text: str) -> str:
    """Day-first dates to ISO (2026-09-08); unrecognised text is returned as is."""
    text = text.strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            pass
    return text


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


def load_scores(source: str) -> dict[str, float]:
    """Manual 1-10 ratings: a CSV with a 'player' (or 'name') and a 'rating' column."""
    df = pd.read_csv(sheets_csv_url(source), dtype=str, keep_default_na=False,
                     encoding="utf-8-sig")
    df.columns = [c.strip().lower() for c in df.columns]
    name_col = next((c for c in ("player", "name") if c in df.columns), None)
    if name_col is None or "rating" not in df.columns:
        raise ValueError(f"need 'player' and 'rating' columns; found {list(df.columns)}")
    scores = {}
    for name, value in zip(df[name_col], df["rating"]):
        name, value = normalize_name(name), value.strip()
        if not name or not value:
            continue
        score = float(value)
        if not 1 <= score <= 10:
            raise ValueError(f"rating for {name} must be 1-10, got {value}")
        scores[name] = score
    return scores


def load_matches(source: str) -> list[Match]:
    df = pd.read_csv(sheets_csv_url(source), header=None, dtype=str,
                     keep_default_na=False, encoding="utf-8-sig")
    rows = [[str(c).strip() for c in row] for row in df.itertuples(index=False)]
    if rows and "team_a" in [c.lower() for c in rows[0]]:
        matches = _parse_table(rows)
    else:
        matches = _parse_blocks(rows)
    for m in matches:
        overlap = set(m.team_a) & set(m.team_b)
        if overlap:
            raise ValueError(f"players on both teams on {m.date}: {sorted(overlap)}")
    return matches


def _parse_table(rows: list[list[str]]) -> list[Match]:
    header = [c.lower() for c in rows[0]]
    missing = [c for c in REQUIRED_COLUMNS if c not in header]
    if missing:
        raise ValueError(f"missing columns {missing}; found {header}")
    col = {name: header.index(name) for name in header}
    matches = []
    for row in rows[1:]:
        if any(not row[col[c]] for c in REQUIRED_COLUMNS):
            continue
        matches.append(
            Match(
                team_a=parse_team(row[col["team_a"]]),
                team_b=parse_team(row[col["team_b"]]),
                goals_a=int(row[col["goals_a"]]),
                goals_b=int(row[col["goals_b"]]),
                date=parse_date(row[col["date"]]) if "date" in col else None,
            )
        )
    return matches


def _parse_blocks(rows: list[list[str]]) -> list[Match]:
    matches = []
    date: str | None = None
    team_a: list[str] = []
    team_b: list[str] = []
    in_block = False
    for line, row in enumerate(rows, 1):
        a, b = row[0], row[1] if len(row) > 1 else ""
        if a.endswith(":"):  # header row: "Whites:,Bibs:,08/09/26"
            if in_block:
                raise ValueError(f"row {line}: new match starts before a score was found")
            in_block, team_a, team_b = True, [], []
            date = parse_date(row[2]) if len(row) > 2 and row[2] else None
        elif not in_block or not (a or b):
            continue
        elif a.isdigit() and b.isdigit():  # score row
            if not team_a or not team_b:
                raise ValueError(f"row {line}: score found but a team has no players")
            matches.append(Match(tuple(team_a), tuple(team_b), int(a), int(b), date))
            in_block = False
        else:  # player row: "1. Hamza,1. Med"
            for name, team in ((a, team_a), (b, team_b)):
                name = normalize_name(name)
                if name:
                    team.append(name)
    if in_block:
        raise ValueError(f"match on {date} has no score row")
    return matches
