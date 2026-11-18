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
  ratings2); empty cell = unrated; a column with no ratings is skipped. Loaded by default if present, or `--ratings`;
  predict/balance use `--version` (default: first manual version, else Unadjusted).
- Cloud sessions cannot reach docs.google.com (network policy): ask the user to upload the CSV.
  The sheet link is NOT stored in the repo (public): it lives in Streamlit secrets / env vars.

## Code

- `qet.py`: entry point. `qet_football/cli.py`: commands `ratings` (`Unadjusted` = matches only,
  shown first and used for sorting, then each version with the manual score in brackets
  e.g. `+0.75 [9]`, Games, Won (draw = 0.5)),
  `evaluate`, `predict`, `balance`.
- `qet_football/data.py`: loads table or block CSV layouts, name clean-up, guests.
- `qet_football/model.py`: ridge regression player ratings on goal difference (alpha=3;
  goal difference clipped to +/-DEFAULT_CAP=4 when fitting, `--cap 0` = off),
  win/draw/loss from a normal on goal difference (sigma floor 2.0). Manual ratings set each
  player's prior mean = beta * (score - RATING_MIDPOINT=5.5); beta (goals per point) fixed at
  DEFAULT_BETA=0.2 (`--beta learned` fits it, pulled to BETA_PRIOR=0.15 / BETA_PENALTY=20);
  rated players' deviation penalty is alpha * DEFAULT_HOLD=20 (`--hold`).
- `qet_football/evaluate.py`: leave-one-out comparison of rating versions (avg goal error,
  outcome right, avg probability of the actual outcome), plus the same on the training set.
- `qet_football/balance.py`: exhaustive 6v6 split search (462 splits), together/apart constraints.
- `app.py`: Streamlit web app (`streamlit run app.py`). Runs locally, and is being shared on
  Streamlit Community Cloud (user's choice) with settings from env vars or Streamlit secrets:
  QET_DATA (sheet link), QET_RATINGS, QET_PASSWORD (simple password gate), QET_VIEWER=true
  (no file paths/saving; manual ratings and match ticks per session only). Links are re-read
  every 5 minutes (+ Reload data button). Deploy steps in README. Display only: reuses the package, no model logic of its own. Tab order (user's
  choice): Ratings (with a "<version> Δ" text column per version = change from the last match
  day, all matches on the latest date; refit without them; "new" = first games that day),
  then Matches, Manual ratings, Balance, Predict, Evaluate. Tabs: Ratings,
  Balance (incl. "most even across all versions"), Predict, Evaluate, Manual ratings (editable
  table, up to 3 versions, edits apply immediately, Save overwrites the ratings file; extra
  file columns beyond 3 would be dropped, with a warning), Matches (Include checkbox per match,
  default ticked; unticked matches are left out of every tab; session only, resets on reload -
  user's choice). Player contribution charts (Altair, diverging blue/red stacked bar per team,
  segment positions computed in Python, net tick, expected score = usual goals per game split
  around the expected difference; text breakdown per team below each chart for phones; in-bar
  names only when they'd fit at phone width) in Predict, Balance and Matches ("Show match" picker; past
  matches use ratings fitted on all included matches, user's choice A, not leave-one-out).
- Tests: `pytest` (fixture in `tests/data/blocks.csv` uses fake names; `tests/test_app.py` uses
  Streamlit's AppTest).

## Status

- Last updated: 2026-11-18.
- Built: ratings, match predictor, team balancer, block-layout loader, manual 1-10 ratings
  as prior (multiple versions), leave-one-out `evaluate` (leave-one-out + training set), goal-difference cap, fixed beta/hold, Streamlit web app with ratings editor, rating midpoint fixed at 5.5, match exclusion and player contribution charts and last-match-day rating change in the app, password + viewer mode for Streamlit Cloud.
  40 tests passing.
- Data: 12 matches (Sep-Nov 2026), 23 players (incl. 10/11 and two games on 17/11, marked A/B
  in a 4th column; 0-7 and 4-3, much lower scoring - asked the user whether they were short
  games, no answer yet). With 12 matches LOO is 3.71 Unadjusted / 3.31 with Hamza 10 / Madhu 1,
  vs 2.58 always-draw. Nas, Nasri, Nasaar are different people (user
  confirmed; sheet later fixed Nasaar->Nas on 6/10, Umer->Umar). Sheet spells Abderrahmane;
  user sometimes types Abdderahmane. Manual ratings: 3 versions, 4-5 players.
- 03/11 (16-10) is a big upset, confirmed legit by the user. With it, leave-one-out is worse
  than always guessing a draw (3.12 with cap 4 vs 2.11; winners 2/9); without it the model
  beats the draw guess (1.47 vs 1.62). Cap 4 added for this; it helps only a little.
- 10/11 (14-10, user will add to sheet): near-rematch of 03/11, same side won again. With it,
  leave-one-out 2.70 vs draw guess 2.30 (Unadjusted). Learned beta collapsed to ~0.05 with
  sparse manual ratings, yet trusting them helped out of sample: Hamza 10 / Madhu 1 at
  beta 0.2, hold 20 gave 2.41 (vs 2.86 learned). User chose 0.2 / 20 as defaults.
- Manual ratings tried on 10 matches (beta 0.2, hold 20): Hamza 10 / Madhu 1 best (LOO 2.41);
  adding Ayman 9 / Akram 2 -> 2.85; Ayman 10 / Madhu 1 -> 2.89 (Ayman's results don't
  support a high rating). Saved data/player_ratings.csv still has older ratings1-3.
- Rating midpoint fixed at 5.5 (was the mean of the given scores, which made an 8 count as below
  average when only strong players were rated). Leave-one-out numbers above were computed
  before this change.
- Treating 1-2 game players as average (guests) in fitting barely changed leave-one-out
  (3.12 -> 3.09/3.02); not implemented. User also declined a `--min-games` filter to hide
  low-game players from the ratings table: keep showing everyone.
- Tried and removed (2026-09-27): a `search` command hill-climbing 1-10 ratings to minimise
  leave-one-out error. It reached ~0.03 error but overfits (a simulation showed it predicts
  new matches worse than honest ratings, and searches from different starts disagreed
  wildly). Don't reintroduce it unless the user asks.
- Next ideas (not agreed yet): keep recording matches; user to compare rating versions with
  `evaluate`; tune alpha once there are ~15+ matches.
