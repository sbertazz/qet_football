# QET Football

Match predictor and team balancer for our 6v6 games.

## How it works

Every player gets a **rating**: how many goals they add to their team's goal
difference compared to an average player. Ratings are fitted on all past
matches at once (ridge regression), so it works even though teams change every
week. With little data, ratings are pulled towards 0 (average); a player with
no history counts as exactly average.

- **Predict**: expected goal difference = sum of team A ratings − sum of team B
  ratings, converted into win / draw / loss probabilities.
- **Balance**: all 462 ways to split 12 players into 6 v 6 are checked, and the
  most even ones are shown.

## Setup (Windows)

```bat
cd C:\python
git clone https://github.com/sbertazz/qet_football.git
cd qet_football
python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]
```

## Match data

One row per match:

| date       | team_a                          | team_b                           | goals_a | goals_b |
|------------|---------------------------------|----------------------------------|---------|---------|
| 2026-01-08 | Marco;Luca;Stefano;Paolo;...    | Davide;Matteo;Simone;Fabio;...   | 7       | 5       |

Players are separated by `;`. Names are case-insensitive (`marco` = `Marco`).
See `data/sample_matches.csv`.

The source can be:
- a local CSV, e.g. `data/matches.csv` (the default; git-ignored), or
- a Google Sheets link, if the sheet is shared as "Anyone with the link can view".

Pass it with `--data`, or set it once with `set QET_DATA=<path or link>`.

## Usage

```bat
qet ratings
qet predict --a "Marco,Luca,Stefano,Paolo,Andrea,Giorgio" --b "Davide,Matteo,Simone,Fabio,Nicola,Alberto"
qet balance "Marco,Luca,Stefano,Paolo,Andrea,Giorgio,Davide,Matteo,Simone,Fabio,Nicola,Enrico"
qet balance "..." --top 5 --together "Marco,Luca" --apart "Andrea,Paolo"
```

`--alpha` (default 3) controls how cautious the ratings are: higher values keep
everyone closer to average, lower values trust the results more.

Try it on the sample data: `qet --data data/sample_matches.csv ratings`.

## Tests

```bat
pytest
```
