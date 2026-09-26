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

The virtual environment lives outside the repo, in `C:\python\pip_envs`.

Once:

```bat
cd C:\python
git clone https://github.com/sbertazz/qet_football.git
python -m venv C:\python\pip_envs\qet_football_env
C:\python\pip_envs\qet_football_env\Scripts\activate
cd C:\python\qet_football
pip install -r requirements.txt
```

Each time:

```bat
C:\python\pip_envs\qet_football_env\Scripts\activate
cd C:\python\qet_football
python qet.py ratings
```

The project itself is not installed: run it from the repo folder with `python qet.py`.

## Match data

Two layouts are accepted, detected automatically.

**Blocks** (how our sheet is written): for each match, a header row with the
two team labels and the date (day/month/year), numbered players, then the score.
Blank rows in between are fine.

```
Whites:,Bibs:,08/09/26
1. Hamza,1. Med,
2. Stefano,2. Alex,
...
6. Hasnain,6. Ayman,
,,
12,12,
```

The left column is team A, the right team B. Numbering, hidden characters and
upper/lower case in names are cleaned up automatically. A name with `+` (e.g.
`Ayman+1`) is a guest: counted as an average player, never rated.

**Table**: one row per match, players separated by `;`
(see `data/sample_matches.csv`):

| date       | team_a                       | team_b                         | goals_a | goals_b |
|------------|------------------------------|--------------------------------|---------|---------|
| 2026-01-08 | Marco;Luca;Stefano;...       | Davide;Matteo;Simone;...       | 7       | 5       |

The source can be:
- a local CSV, e.g. `data/matches.csv` (the default; git-ignored), or
- a Google Sheets link, if the sheet is shared as "Anyone with the link can view".

Pass it with `--data` (before the command, e.g. `python qet.py --data <link> ratings`), or set it once with `set QET_DATA=<path or link>`.

## Usage

```bat
python qet.py ratings
python qet.py predict --a "Marco,Luca,Stefano,Paolo,Andrea,Giorgio" --b "Davide,Matteo,Simone,Fabio,Nicola,Alberto"
python qet.py balance "Marco,Luca,Stefano,Paolo,Andrea,Giorgio,Davide,Matteo,Simone,Fabio,Nicola,Enrico"
python qet.py balance "..." --top 5 --together "Marco,Luca" --apart "Andrea,Paolo"
```

`--alpha` (default 3) controls how cautious the ratings are: higher values keep
everyone closer to average, lower values trust the results more.

Try it on the sample data: `python qet.py --data data/sample_matches.csv ratings`.

## Tests

```bat
pytest
```
