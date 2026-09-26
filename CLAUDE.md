# QET Football

Match predictor and team balancer for a 6v6 friends' football group (QET is the
group's name). 20-25 people rotate; teams change every game. See README.md.

## Working rules

- Always describe the planned change and wait for the user's confirmation before writing or changing code.
- Real match data (`data/matches.csv`) contains friends' names and is git-ignored: never commit it.
- At the end of a working session, update the "Status" section below.

## User's setup (Windows)

- Repo: `C:\python\qet_football`
- Virtual env (outside the repo): `C:\python\pip_envs\qet_football_env`
  - activate: `C:\python\pip_envs\qet_football_env\Scripts\activate`
- The project is NOT installed as a package: `pip install -r requirements.txt`, then run `python qet.py <command>` from the repo root.

## Data

- Source of truth: the group's Google Sheet, written in "block" layout per match: header row
  (two team labels + date dd/mm/yy), 6 numbered player rows ("1. Name"), then the score row.
  Pasted names contain invisible characters (U+2060); the loader strips them.
- Guests are written as `Name+1`: they count as an average player and are never rated.
- On the user's machine: `data\matches.csv` (CSV export) or the sheet link via `--data` / `QET_DATA`.
- Cloud sessions cannot reach docs.google.com (network policy): ask the user to upload the CSV.

## Code

- `qet.py`: entry point. `qet_football/cli.py`: commands `ratings`, `predict`, `balance`.
- `qet_football/data.py`: loads table or block CSV layouts, name clean-up, guests.
- `qet_football/model.py`: ridge regression player ratings on goal difference (alpha=3),
  win/draw/loss from a normal on goal difference (sigma floor 2.0).
- `qet_football/balance.py`: exhaustive 6v6 split search (462 splits), together/apart constraints.
- Tests: `pytest` (fixture in `tests/data/blocks.csv` uses fake names).

## Status

- Last updated: 2026-09-26.
- Built: ratings, match predictor, team balancer, block-layout loader. 12 tests passing.
- Data: 4 matches (Sept 2026), ~15 rated players: too little for meaningful ratings yet.
- Next ideas (not agreed yet): keep recording matches; check prediction accuracy once there
  is more data (e.g. leave-one-out) and tune alpha.
