"""QET Football web app.

    streamlit run app.py

A thin UI over the qet_football package: ratings, team balancing, match
predictions, leave-one-out evaluation and a check of the loaded matches.
Runs locally; data is read from the same files as the command line tool.
"""

from __future__ import annotations

import difflib
import os
from collections import Counter

import pandas as pd
import streamlit as st

from qet_football.balance import balance_teams
from qet_football.cli import DEFAULT_DATA, DEFAULT_RATINGS, NO_RATINGS
from qet_football.data import is_guest, load_matches, load_score_versions, normalize_name
from qet_football.evaluate import in_sample, leave_one_out, summarize
from qet_football.model import DEFAULT_BETA, DEFAULT_CAP, DEFAULT_HOLD, RatingModel

st.set_page_config(page_title="QET Football", page_icon="⚽", layout="wide")


# --- loading -------------------------------------------------------------------

def _stamp(source: str) -> float | None:
    """Modification time for local files, so edits are picked up on rerun."""
    return os.path.getmtime(source) if os.path.exists(source) else None


@st.cache_data(show_spinner=False)
def _load_matches(source: str, stamp: float | None):
    return load_matches(source)


@st.cache_data(show_spinner=False)
def _load_versions(source: str, stamp: float | None):
    return load_score_versions(source)


def _names(text: str) -> list[str]:
    return [normalize_name(p) for p in text.replace(";", ",").split(",") if p.strip()]


def _signed(x: float) -> str:
    return f"{x:+.2f}"


# --- sidebar -------------------------------------------------------------------

st.sidebar.title("⚽ QET Football")
data_source = st.sidebar.text_input(
    "Match data (CSV path or Google Sheet link)",
    os.environ.get("QET_DATA") or DEFAULT_DATA,
)
ratings_source = st.sidebar.text_input(
    "Manual ratings CSV (leave empty for none)",
    os.environ.get("QET_RATINGS") or (DEFAULT_RATINGS if os.path.exists(DEFAULT_RATINGS) else ""),
)

try:
    matches = _load_matches(data_source, _stamp(data_source))
except Exception as e:  # shown to the user, not raised
    st.error(f"Could not load matches from `{data_source}`: {e}")
    st.stop()
try:
    user_versions = _load_versions(ratings_source, _stamp(ratings_source)) if ratings_source else {}
except Exception as e:
    st.error(f"Could not load manual ratings from `{ratings_source}`: {e}")
    st.stop()

versions = {NO_RATINGS: {}, **user_versions}
default_version = 1 if user_versions else 0
version = st.sidebar.selectbox(
    "Ratings version for Balance / Predict", list(versions), index=default_version
)

with st.sidebar.expander("Model settings"):
    alpha = st.slider("Alpha: how strongly players are pulled to their starting point",
                      0.5, 30.0, 3.0, 0.5)
    cap = st.slider("Cap: max goal difference per match when fitting (0 = off)",
                    0, 10, int(DEFAULT_CAP))
    learn_beta = st.checkbox("Learn goals per rating point from the matches", value=False)
    beta = None if learn_beta else st.slider(
        "Beta: goals per manual rating point", 0.0, 0.5, DEFAULT_BETA, 0.05
    )
    hold = st.slider("Hold: how much more firmly rated players stay at your rating",
                     1.0, 50.0, DEFAULT_HOLD, 1.0)
options = {"alpha": alpha, "cap": cap, "beta": beta, "hold": hold}

st.sidebar.caption(
    f"{len(matches)} matches · "
    + (f"rating versions: {', '.join(user_versions)}" if user_versions else "no manual ratings")
)

models = {v: RatingModel(**options).fit(matches, s) for v, s in versions.items()}
known_players = sorted({p for m in models.values() for p in m.ratings})


def _prediction_rows(team_a: list[str], team_b: list[str]) -> pd.DataFrame:
    rows = []
    for v, model in models.items():
        p = model.predict(team_a, team_b)
        rows.append({
            "Version": v,
            "Expected diff (A - B)": f"{p.goal_diff:+.2f}",
            "A wins": f"{p.p_win_a:.0%}",
            "Draw": f"{p.p_draw:.0%}",
            "B wins": f"{p.p_win_b:.0%}",
        })
    return pd.DataFrame(rows)


def _player_picker(label: str, key: str) -> list[str]:
    picked = st.multiselect(label, known_players, key=f"{key}_known")
    extra = st.text_input("Other players, comma separated (new players count as average)",
                          key=f"{key}_extra")
    players = list(dict.fromkeys([*picked, *_names(extra)]))
    unknown = [p for p in players if p not in known_players and not is_guest(p)]
    if unknown:
        st.caption(f"No history (counted as average): {', '.join(unknown)}")
    return players


tab_ratings, tab_balance, tab_predict, tab_eval, tab_matches = st.tabs(
    ["Ratings", "Balance", "Predict", "Evaluate", "Matches"]
)

# --- ratings -------------------------------------------------------------------

with tab_ratings:
    unadjusted = models[NO_RATINGS]
    table = pd.DataFrame(
        [
            {
                "Player": p,
                **{v: round(m.rating(p), 2) for v, m in models.items()},
                "Games": unadjusted.games.get(p, 0),
                "Won": unadjusted.won.get(p, 0.0),
                "Your ratings": " · ".join(
                    f"{v} {s[p]:g}" for v, s in user_versions.items() if p in s
                ),
            }
            for p in known_players
        ]
    ).sort_values(NO_RATINGS, ascending=False, ignore_index=True)
    table.index += 1

    def _highlight(column: pd.Series) -> list[str]:
        rated = user_versions.get(column.name, {})
        return ["background-color: rgba(255, 200, 0, 0.25)" if p in rated else ""
                for p in table.loc[column.index, "Player"]]

    styled = table.style.apply(_highlight, subset=list(user_versions)).format(
        {v: _signed for v in models} | {"Won": "{:.1f}"}
    )
    st.dataframe(styled, width="stretch", height=min(35 * (len(table) + 1) + 3, 900))
    notes = [f"**{v}**: {len(s)} players rated, 1 point = {models[v].beta:.2f} goals"
             for v, s in user_versions.items()]
    st.caption(
        "Rating = goals a player adds to their team's goal difference per game versus an "
        "average player. Won counts a draw as 0.5. Highlighted cells = you rated that player "
        "in that version. Click a column header to sort."
        + ("  \n" + "  \n".join(notes) if notes else "")
    )

# --- balance -------------------------------------------------------------------

with tab_balance:
    players = _player_picker("Players (pick an even number, usually 12)", "balance")
    col1, col2, col3 = st.columns(3)
    together = col1.multiselect("Keep together", players)
    apart = col2.multiselect("Keep apart (2 players)", players, max_selections=2)
    top = col3.slider("Options to show", 1, 10, 3)
    criterion = st.radio(
        "Balance using",
        [f"Selected version ({version})", "All versions: most even in the worst case"],
        horizontal=True,
    )

    if len(players) < 2 or len(players) % 2:
        st.info(f"Pick an even number of players ({len(players)} selected).")
    elif len(players) > 18:
        st.warning("Too many players to try every split; pick at most 18.")
    else:
        splits = balance_teams(
            players, models[version], top=100_000,
            together=[together] if together else [], apart=[apart] if len(apart) == 2 else [],
        )
        if not splits:
            st.warning("No split satisfies the constraints.")

        def _gap(split) -> float:
            if criterion.startswith("Selected"):
                return abs(split.goal_diff)
            return max(abs(m.strength(split.team_a) - m.strength(split.team_b))
                       for m in models.values())

        for i, s in enumerate(sorted(splits, key=_gap)[:top], 1):
            st.subheader(f"Option {i}")
            a, b, preds = st.columns([1, 1, 2])
            a.markdown("**Team A**  \n" + "  \n".join(s.team_a))
            b.markdown("**Team B**  \n" + "  \n".join(s.team_b))
            preds.dataframe(_prediction_rows(list(s.team_a), list(s.team_b)),
                            hide_index=True, width="stretch")

# --- predict -------------------------------------------------------------------

with tab_predict:
    left, right = st.columns(2)
    with left:
        team_a = _player_picker("Team A", "pred_a")
    with right:
        team_b = _player_picker("Team B", "pred_b")
    both = set(team_a) & set(team_b)
    if both:
        st.warning(f"On both teams: {', '.join(sorted(both))}")
    elif team_a and team_b:
        st.dataframe(_prediction_rows(team_a, team_b), hide_index=True,
                     width="stretch")
        st.caption("Win / draw / loss assume the real goal difference varies by about "
                   f"±{models[version].sigma:.1f} goals around the expectation.")
    else:
        st.info("Pick players for both teams.")

# --- evaluate ------------------------------------------------------------------

with tab_eval:
    if len(matches) < 2:
        st.info("Need at least 2 matches to evaluate.")
    else:
        loo = leave_one_out(matches, versions, **options)
        train = in_sample(matches, versions, **options)
        per_match = pd.DataFrame(
            [
                {
                    "Match": m.date or f"#{i + 1}",
                    "Score": f"{m.goals_a}-{m.goals_b}",
                    "Actual": f"{m.goal_diff:+d}",
                    **{v: f"{loo[v][i].goal_diff:+.1f}" for v in versions},
                }
                for i, m in enumerate(matches)
            ]
        )
        st.markdown("**Each match predicted from all the others** "
                    "(goal difference, left − right)")
        st.dataframe(per_match, hide_index=True, width="stretch")

        def _summary(preds) -> pd.DataFrame:
            rows = {}
            for v, p in preds.items():
                s = summarize(matches, p)
                rows[v] = {"Avg error (goals)": round(s.mean_error, 2),
                           "Outcome right": s.correct,
                           "Avg P(actual)": round(s.mean_p_actual, 3)}
            return pd.DataFrame(rows)

        def _best(row: pd.Series) -> list[str]:
            best = row.min() if row.name == "Avg error (goals)" else row.max()
            return ["font-weight: bold; background-color: rgba(0, 180, 0, 0.2)"
                    if x == best else "" for x in row]

        draw_guess = sum(abs(m.goal_diff) for m in matches) / len(matches)
        left, right = st.columns(2)
        for col, title, preds in ((left, "Leave-one-out (unseen matches)", loo),
                                  (right, "Training set (matches it was fitted on)", train)):
            col.markdown(f"**{title}**")
            summary = _summary(preds)
            n = len(matches)
            styled = (
                summary.style.apply(_best, axis=1)
                .format("{:.2f}", subset=pd.IndexSlice[["Avg error (goals)"], :])
                .format(lambda x: f"{x:.0f}/{n}", subset=pd.IndexSlice[["Outcome right"], :])
                .format("{:.0%}", subset=pd.IndexSlice[["Avg P(actual)"], :])
            )
            col.dataframe(styled, width="stretch")
        st.caption(
            f"Always guessing a draw would give an average error of {draw_guess:.2f} goals. "
            f"Outcome right is out of {len(matches)}. Green = best. Trust leave-one-out over "
            "the training set; a big gap between them means the model memorises matches."
            + (f" Only {len(matches)} matches: differences may just be luck."
               if len(matches) < 15 else "")
        )

# --- matches -------------------------------------------------------------------

with tab_matches:
    st.dataframe(
        pd.DataFrame(
            [
                {"Date": m.date, "Left team": ", ".join(m.team_a),
                 "Score": f"{m.goals_a}-{m.goals_b}", "Right team": ", ".join(m.team_b)}
                for m in matches
            ]
        ),
        hide_index=True, width="stretch",
    )
    games = Counter(p for m in matches for p in (*m.team_a, *m.team_b) if not is_guest(p))
    similar = sorted(
        {tuple(sorted((p, q))) for p in games
         for q in difflib.get_close_matches(p, games, n=3, cutoff=0.8) if q != p}
    )
    if similar:
        st.warning("Similar names, check they're really different people: "
                   + "; ".join(f"{p} / {q}" for p, q in similar))
    st.caption("Players and games: "
               + ", ".join(f"{p} ({n})" for p, n in sorted(games.items())))
