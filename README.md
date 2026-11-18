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

## Web app

```bat
streamlit run app.py
```

Opens in your browser at http://localhost:8501 and runs only on your machine. Tabs:
**Ratings** (sortable; cells you rated are highlighted; a Δ column per version shows the
change from the last match day, "new" for players whose first games were that day),
**Player** (pick a player: how much each of his matches raised or lowered his rating -
his rating minus his rating without that match - with the expected result, plus the
biggest effects from matches he didn't play), **Balance** (pick players,
together/apart, balance on one version or on all at once), **Predict**, **Evaluate**
(leave-one-out and training set), **Manual ratings** (edit up to 3 versions in a
table; edits apply at once, **Save** writes `data/player_ratings.csv`) and **Matches** (untick a match to leave it out of everything for this session; it resets
on reload) (what was loaded, with a warning for
similar names). Predict, Balance and Matches also show a **player contribution chart**: one stacked bar
per team, each player a segment (blue adds goals, red costs goals versus an average
player), with the team total marked and an expected score. Underneath, the same numbers as text (one
line per team, best player first) for small screens. For past matches the ratings
come from all included matches, that one too.
The sidebar sets the data files, the ratings version and the model
settings (alpha, cap, beta, hold).

### Sharing it online (Streamlit Community Cloud)

1. Make sure the Google Sheet is shared as **Anyone with the link can view**.
2. Go to https://share.streamlit.io, sign in with GitHub, click **Create app** and pick
   repo `sbertazz/qet_football`, branch `main`, main file `app.py`.
3. Under **Advanced settings**, choose Python 3.11 or newer and paste into **Secrets**:

   ```toml
   QET_DATA = "https://docs.google.com/spreadsheets/d/<sheet id>/edit?gid=0"
   QET_PASSWORD = "<pick a password>"
   QET_VIEWER = "true"
   ```

4. Deploy, then share the app link and the password with the group.

The shared app reads the sheet live (re-read every 5 minutes, or with **Reload data**),
asks for the password first, and runs in viewer mode: no file paths and no saving.
Anyone can still untick matches and enter manual ratings for their own session. Pushes
to `main` redeploy it automatically. The same settings can go in a local, git-ignored
`.streamlit/secrets.toml` or in environment variables.

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

## Manual ratings (optional)

`data/player_ratings.csv` (git-ignored) gives your own 1-10 rating of players.
It can hold several versions, one column each (any column names):

```
player,ratings1,ratings2,ratings3
Hamza,9,8,9
Madhu,3,4,
Alex,7,7,6
```

An empty cell means "not rated in that version". A single `rating` column works too.

Ratings set each player's starting point: above or below average according to
your rating. Match results then adjust from there, and as more games are
recorded they take over. One rating point is worth 0.2 goals (`--beta`, or
`--beta learned` to estimate it from the matches), and rated players are held
at your rating 20 times more firmly than unrated ones (`--hold`). Players you
don't rate start as average, i.e. a 5.5 on your 1-10 scale. Use another file with
`--ratings <path>`.

- `ratings` shows the model's ratings for `Unadjusted` (match results only, sorted by it)
  and for every version, then games played and games won (a draw counts 0.5). In the
  version columns your manual rating is shown in brackets, e.g. `+0.75 [9]`.
- `evaluate` predicts each past match from all the others, for every version and
  `Unadjusted`, and shows which version predicts best. It also shows the error on the
  training set (fitted and scored on all matches): a big gap between the two means overfitting. With few matches the winner may
  just be luck.
- `predict` and `balance` use the first version, or pick one with `--version ratings2`.

## Usage

```bat
python qet.py ratings
python qet.py evaluate
python qet.py --version ratings2 predict --a "Marco,Luca,Stefano,Paolo,Andrea,Giorgio" --b "Davide,Matteo,Simone,Fabio,Nicola,Alberto"
python qet.py balance "Marco,Luca,Stefano,Paolo,Andrea,Giorgio,Davide,Matteo,Simone,Fabio,Nicola,Enrico"
python qet.py balance "..." --top 5 --together "Marco,Luca" --apart "Andrea,Paolo"
```

`--cap` (default 4) clips each match's goal difference to +/-4 goals when fitting, so one
blowout can't dominate the ratings (`--cap 0` turns it off; `evaluate` still scores
predictions against the real results).

`--alpha` (default 3) controls how cautious the ratings are: higher values keep
everyone closer to average, lower values trust the results more.

Try it on the sample data: `python qet.py --data data/sample_matches.csv ratings`.

## Tests

```bat
pytest
```
