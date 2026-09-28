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
- Manual 1-10 ratings from the user: `data/player_ratings.csv`, git-ignored (opinions about
  friends; the repo is public). Columns: `player` + one column per version (e.g. ratings1,
  ratings2); empty cell = unrated. Loaded by default if present, or `--ratings`;
  predict/balance use `--version` (default: first manual version, else Unadjusted).
- Cloud sessions cannot reach docs.google.com (network policy): ask the user to upload the CSV.

## Code

- `qet.py`: entry point. `qet_football/cli.py`: commands `ratings` (`Unadjusted` = matches only,
  shown first and used for sorting, then each version with the manual score in brackets
  e.g. `+0.75 [9]`, Games, Won (draw = 0.5)),
  `evaluate`, `predict`, `balance`.
- `qet_football/data.py`: loads table or block CSV layouts, name clean-up, guests.
- `qet_football/model.py`: ridge regression player ratings on goal difference (alpha=3),
  win/draw/loss from a normal on goal difference (sigma floor 2.0). Manual ratings set each
  player's prior mean = beta * (score - mean score); beta (goals per point) is fitted jointly,
  pulled towards BETA_PRIOR=0.15 with BETA_PENALTY=20.
- `qet_football/evaluate.py`: leave-one-out comparison of rating versions (avg goal error,
  outcome right, avg probability of the actual outcome).
- `qet_football/balance.py`: exhaustive 6v6 split search (462 splits), together/apart constraints.
- Tests: `pytest` (fixture in `tests/data/blocks.csv` uses fake names).

## Status

- Last updated: 2026-09-28.
- Built: ratings, match predictor, team balancer, block-layout loader, manual 1-10 ratings
  as prior (multiple versions), leave-one-out `evaluate`. 23 tests passing.
- Data: 4 matches (Sept 2026), ~15 players; 3 manual rating versions (4-5 players each;
  ratings1 fitted beta ~0.23).
  Too little for meaningful ratings yet.
- Leave-one-out on 4 matches: user's ratings1 0.90 goals avg error, ratings2/3 (Ayman=8) 1.20,
  Unadjusted 1.29.
- Tried and removed (2026-09-27): a `search` command hill-climbing 1-10 ratings to minimise
  leave-one-out error. It reached ~0.03 error but overfits (a simulation showed it predicts
  new matches worse than honest ratings, and searches from different starts disagreed
  wildly). Don't reintroduce it unless the user asks.
- Next ideas (not agreed yet): keep recording matches; user to compare rating versions with
  `evaluate`; tune alpha once there are ~15+ matches.
