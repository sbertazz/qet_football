"""QET Football web app.

    streamlit run app.py

A thin UI over the qet_football package: ratings, team balancing, match
predictions, leave-one-out evaluation, editing manual ratings (up to 3
versions) and a check of the loaded matches.
Runs locally; data is read from the same files as the command line tool.

Settings, from environment variables or Streamlit secrets (.streamlit/secrets.toml
locally, the app's Secrets on Streamlit Community Cloud):
    QET_DATA      match data: CSV path or Google Sheet link
    QET_RATINGS   manual ratings file
    QET_PASSWORD  if set, the app asks for this password first
    QET_VIEWER    "true" for the shared version: no file paths, no saving;
                  manual ratings and match ticks work for the session only
"""

from __future__ import annotations

import difflib
import hmac
import os
import time
from collections import Counter

import altair as alt
import pandas as pd
import streamlit as st

from qet_football.balance import balance_teams
from qet_football.cli import DEFAULT_DATA, DEFAULT_RATINGS, NO_RATINGS
from qet_football.data import is_guest, load_matches, load_score_versions, normalize_name
from qet_football.evaluate import in_sample, leave_one_out, summarize
from qet_football.model import DEFAULT_BETA, DEFAULT_CAP, DEFAULT_HOLD, RatingModel

st.set_page_config(page_title="QET Football", page_icon="⚽", layout="wide")


def _setting(name: str, default: str = "") -> str:
    """Environment variable, else Streamlit secret, else default."""
    if os.environ.get(name):
        return os.environ[name]
    try:
        return str(st.secrets.get(name, default))
    except Exception:  # no secrets file at all
        return default


VIEWER = _setting("QET_VIEWER").strip().lower() in ("1", "true", "yes")
PASSWORD = _setting("QET_PASSWORD")

if PASSWORD and not st.session_state.get("authenticated"):
    st.title("⚽ QET Football")
    entered = st.text_input("Password", type="password", key="password")
    if entered and hmac.compare_digest(entered, PASSWORD):
        st.session_state["authenticated"] = True
        st.rerun()
    if entered:
        st.error("Wrong password.")
    st.stop()


# --- loading -------------------------------------------------------------------

SHEET_REFRESH_SECONDS = 300


def _stamp(source: str) -> float | None:
    """Cache key: modification time for local files (edits are picked up on rerun);
    a 5-minute time slot for links, so a shared sheet is re-read regularly."""
    if os.path.exists(source):
        return os.path.getmtime(source)
    if source.startswith("http"):
        return time.time() // SHEET_REFRESH_SECONDS
    return None


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
if VIEWER:
    # Shared version: the data source is fixed by the settings and never shown.
    data_source = _setting("QET_DATA", DEFAULT_DATA)
    ratings_source = _setting("QET_RATINGS")
else:
    data_source = st.sidebar.text_input(
        "Match data (CSV path or Google Sheet link)", _setting("QET_DATA", DEFAULT_DATA)
    )
    ratings_source = st.sidebar.text_input(
        "Manual ratings file (loaded at start; 'Save' writes here)",
        _setting("QET_RATINGS", DEFAULT_RATINGS),
    )
if data_source.startswith("http") and st.sidebar.button("Reload data"):
    _load_matches.clear()

try:
    all_matches = _load_matches(data_source, _stamp(data_source))
except Exception as e:  # shown to the user, not raised
    where = "the match data" if VIEWER else f"`{data_source}`"
    hint = (" Check the Google Sheet is shared as 'Anyone with the link can view'."
            if data_source.startswith("http") else "")
    st.error(f"Could not load matches from {where}: {e}.{hint}")
    st.stop()
try:
    file_versions = (
        _load_versions(ratings_source, _stamp(ratings_source))
        if ratings_source and os.path.exists(ratings_source) else {}
    )
except Exception as e:
    st.error(f"Could not load manual ratings from `{ratings_source}`: {e}")
    st.stop()

match_players = sorted(
    {p for m in all_matches for p in (*m.team_a, *m.team_b) if not is_guest(p)}
)

tab_ratings, tab_matches, tab_manual, tab_balance, tab_predict, tab_eval = st.tabs(
    ["Ratings", "Matches", "Manual ratings", "Balance", "Predict", "Evaluate"]
)

# --- matches -------------------------------------------------------------------
# Rendered before the model (its tab is further right) so the Include ticks apply
# to every other tab on the same run. Exclusions last for this browser session.

with tab_matches:
    picked = st.data_editor(
        pd.DataFrame(
            [
                {"Include": True, "Date": m.date, "Left team": ", ".join(m.team_a),
                 "Score": f"{m.goals_a}-{m.goals_b}", "Right team": ", ".join(m.team_b)}
                for m in all_matches
            ]
        ),
        key="match_picker",
        hide_index=True,
        width="stretch",
        height=min(35 * (len(all_matches) + 1) + 3, 900),
        disabled=["Date", "Left team", "Score", "Right team"],
        column_config={"Include": st.column_config.CheckboxColumn(
            "Include", help="Untick to leave this match out of every tab", default=True)},
    )
    matches = [m for m, keep in zip(all_matches, picked["Include"]) if keep]
    n_excluded = len(all_matches) - len(matches)
    st.caption(
        "Untick a match to leave it out of ratings, balance, predictions and evaluation. "
        "Exclusions reset when you reload the page."
        + (f" **{n_excluded} excluded.**" if n_excluded else "")
    )
    games = Counter(p for m in all_matches for p in (*m.team_a, *m.team_b) if not is_guest(p))
    similar = sorted(
        {tuple(sorted((p, q))) for p in games
         for q in difflib.get_close_matches(p, games, n=3, cutoff=0.8) if q != p}
    )
    if similar:
        st.warning("Similar names, check they're really different people: "
                   + "; ".join(f"{p} / {q}" for p, q in similar))
    st.caption("Players and games (all matches): "
               + ", ".join(f"{p} ({n})" for p, n in sorted(games.items())))

excluded_note = (
    f"{n_excluded} of {len(all_matches)} matches excluded (see the Matches tab)."
    if n_excluded else ""
)

# --- manual ratings ------------------------------------------------------------
# Also rendered before the model, so edits apply to every other tab on the same run.

MAX_VERSIONS = 3
SLOTS = [f"v{i + 1}" for i in range(MAX_VERSIONS)]


def _save_ratings(path: str, versions: dict[str, dict[str, float]]) -> None:
    players = sorted({p for scores in versions.values() for p in scores})
    lines = [",".join(["player", *versions])]
    for p in players:
        lines.append(",".join([p, *(f"{versions[v][p]:g}" if p in versions[v] else ""
                                    for v in versions)]))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _ratings_editor() -> dict[str, dict[str, float]]:
    """Editable table of up to MAX_VERSIONS rating versions; returns the non-empty ones."""
    file_names = list(file_versions)
    if len(file_names) > MAX_VERSIONS:
        st.warning(
            f"`{ratings_source}` has {len(file_names)} rating versions; only the first "
            f"{MAX_VERSIONS} are shown here, and saving would drop: "
            + ", ".join(file_names[MAX_VERSIONS:])
        )
    defaults = [
        file_names[i] if i < len(file_names)
        else next(n for n in (f"ratings{i + 1}", f"version{i + 1}") if n not in file_names)
        for i in range(MAX_VERSIONS)
    ]

    keep_note = ("Ratings here are just for you and are lost when you reload the page."
                 if VIEWER else "**Save** keeps them for next time.")
    st.markdown(
        "Rate players from **1 to 10**; **5.5 = average**, and an empty cell (shown as "
        "*None*) means not rated (counted as average). To clear a cell, select it and press "
        "Delete. Edits apply to every tab straight away. " + keep_note + " "
        "A version with no ratings is ignored. Add a player who isn't "
        "in the matches with the + row at the bottom."
    )
    name_cols = st.columns(MAX_VERSIONS)
    names = [name_cols[i].text_input(f"Version {i + 1} name", defaults[i], key=f"vname{i}").strip()
             for i in range(MAX_VERSIONS)]

    players = sorted(set(match_players) | {p for sc in file_versions.values() for p in sc})
    base = pd.DataFrame({
        "Player": players,
        **{slot: pd.Series([file_versions[file_names[i]].get(p) if i < len(file_names) else None
                            for p in players], dtype="float")
           for i, slot in enumerate(SLOTS)},
    })
    edited = st.data_editor(
        base,
        key="ratings_editor",
        num_rows="dynamic",
        hide_index=True,
        width="stretch",
        height=min(35 * (len(base) + 2) + 3, 900),
        column_config={
            "Player": st.column_config.TextColumn("Player", required=True),
            **{slot: st.column_config.NumberColumn(names[i] or f"Version {i + 1}",
                                                   min_value=1, max_value=10, step=1)
               for i, slot in enumerate(SLOTS)},
        },
    )

    versions: dict[str, dict[str, float]] = {}
    for i, slot in enumerate(SLOTS):
        scores = {
            normalize_name(str(row["Player"])): float(row[slot])
            for _, row in edited.iterrows()
            if pd.notna(row["Player"]) and str(row["Player"]).strip() and pd.notna(row[slot])
        }
        if not scores:
            continue
        name = names[i] or f"Version {i + 1}"
        if name == NO_RATINGS or name in versions:
            st.warning(f"Version name '{name}' is already used; version {i + 1} is ignored.")
            continue
        versions[name] = scores

    summary = ", ".join(f"{v} ({len(sc)} players)" for v, sc in versions.items())
    st.caption(f"In use: {summary}" if summary else "No manual ratings in use.")
    if not VIEWER and st.button("Save to file", disabled=not ratings_source):
        _save_ratings(ratings_source, versions)
        st.success(f"Saved to `{ratings_source}`.")
    return versions


with tab_manual:
    user_versions = _ratings_editor()

# --- model -----------------------------------------------------------------------

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
    (f"{len(matches)} of {len(all_matches)} matches used ({n_excluded} excluded) · "
     if n_excluded else f"{len(matches)} matches · ")
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


# Contribution charts: diverging blue (adds goals) / red (costs goals), from the
# dataviz reference palette; labels inside segments are white.
POSITIVE, NEGATIVE = "#2a78d6", "#e34948"
# Assumed plot width (px) when deciding whether a name fits inside its segment. Kept
# phone-sized: the app can't see the screen width, and every name is also listed in
# text under the chart.
CHART_WIDTH = 300
GOALS_PER_GAME = (
    sum(m.goals_a + m.goals_b for m in matches) / len(matches) if matches else 25.0
)


def _match_summary(team_a, team_b, model, names=("Team A", "Team B"), actual=None) -> str:
    """One line: expected winner, margin, rough score, win/draw/loss, actual score."""
    p = model.predict(team_a, team_b)
    a_goals = GOALS_PER_GAME / 2 + p.goal_diff / 2
    b_goals = GOALS_PER_GAME / 2 - p.goal_diff / 2
    if abs(p.goal_diff) < 0.5:
        head = "Expected: even game"
    else:
        leader = names[0] if p.goal_diff > 0 else names[1]
        head = f"Expected: **{leader} by {abs(p.goal_diff):.1f}**"
    text = (f"{head} (≈ {round(a_goals)}–{round(b_goals)}) · {names[0]} {p.p_win_a:.0%} · "
            f"draw {p.p_draw:.0%} · {names[1]} {p.p_win_b:.0%}")
    if actual is not None:
        text += f" · **Actual: {actual[0]}–{actual[1]}**"
    return text


def _contribution_chart(team_a, team_b, model, names=("Team A", "Team B"), height=150):
    """Stacked bar per team: each player's rating is a segment growing out from zero
    (positives to the right, negatives to the left), so the team total is where the
    two sides net out. Segment positions are computed here, not by Vega's stack."""
    nets = [sum(model.rating(p) for p in t) for t in (team_a, team_b)]
    names = [f"{n} · net {t:+.2f}" for n, t in zip(names, nets)]
    rows = []
    for team, players in zip(names, (team_a, team_b)):
        rated = [(p, model.rating(p)) for p in players]
        for sign in (1, -1):
            edge = 0.0
            side = sorted((pr for pr in rated if (pr[1] > 0 if sign > 0 else pr[1] < 0)),
                          key=lambda pr: -abs(pr[1]))  # biggest contributors nearest zero
            for p, r in side:
                rows.append({"Team": team, "Player": p, "Rating": r, "start": edge,
                             "end": edge + r, "mid": edge + r / 2,
                             "Effect": "adds goals" if r > 0 else "costs goals",
                             "Shown": f"{r:+.2f}"})
                edge += r
    df = pd.DataFrame(rows, columns=["Team", "Player", "Rating", "start", "end", "mid",
                                     "Effect", "Shown"])
    totals = pd.DataFrame({"Team": names, "Total": nets})
    totals["Text"] = totals.Total.map(lambda t: f"{t:+.2f}")
    extent = max([abs(v) for v in (*df.start, *df.end)] + [0.5]) * 1.1
    px_per_goal = CHART_WIDTH / (2 * extent)
    df["Label"] = [p if abs(r) * px_per_goal > 7 * len(p) + 12 else ""
                   for p, r in zip(df.Player, df.Rating)]

    x_scale = alt.Scale(domain=[-extent, extent])
    y = alt.Y("Team:N", sort=list(names), title=None,
              scale=alt.Scale(paddingInner=0.35, paddingOuter=0.2),
              axis=alt.Axis(labelFontSize=13, labelFontWeight="bold", ticks=False, domain=False,
                            labelLimit=260))
    base = alt.Chart(df).encode(y=y)
    bars = base.mark_bar(stroke="white", strokeWidth=2, cornerRadius=3).encode(
        x=alt.X("start:Q", scale=x_scale,
                title="Goals added per game vs an average player",
                axis=alt.Axis(tickCount=8, gridOpacity=0.4)),
        x2="end:Q",
        color=alt.Color("Effect:N", scale=alt.Scale(domain=["adds goals", "costs goals"],
                                                    range=[POSITIVE, NEGATIVE]),
                        legend=alt.Legend(title=None, orient="top")),
        tooltip=[alt.Tooltip("Team:N"), alt.Tooltip("Player:N"),
                 alt.Tooltip("Shown:N", title="Rating (goals/game)")],
    )
    names_in_bars = base.transform_filter(alt.datum.Label != "").mark_text(
        color="white", fontSize=12, fontWeight=500
    ).encode(x=alt.X("mid:Q", scale=x_scale), text="Label:N")
    zero = alt.Chart(pd.DataFrame({"x": [0.0]})).mark_rule(color="gray").encode(
        x=alt.X("x:Q", scale=x_scale))
    net = alt.Chart(totals).encode(y=y, x=alt.X("Total:Q", scale=x_scale))
    net_tick = net.mark_tick(color="black", thickness=3, size=height / 2.6).encode(
        tooltip=[alt.Tooltip("Team:N"), alt.Tooltip("Text:N", title="Team total")])
    return (bars + names_in_bars + zero + net_tick).properties(height=height)


def _contribution_text(team_a, team_b, model, names=("Team A", "Team B")) -> None:
    """The chart's numbers as text, readable on a phone: one line per team, best first.
    Columns sit side by side on a wide screen and stack on a narrow one."""
    for col, name, team in zip(st.columns(2), names, (team_a, team_b)):
        def entry(p: str) -> str:
            known = normalize_name(p) in model.ratings and not is_guest(p)
            r = model.rating(p)
            value = (f":green[{r:+.2f}]" if r > 0.005 else f":red[{r:+.2f}]" if r < -0.005
                     else "0.00")
            return f"{p} {value}" + ("" if known else " (new)")

        ranked = sorted(team, key=model.rating, reverse=True)
        net = sum(model.rating(p) for p in team)
        col.markdown(f"**{name} · net {net:+.2f}**  \n" + " · ".join(entry(p) for p in ranked))


def _show_contributions(team_a, team_b, model, names=("Team A", "Team B"), actual=None,
                        height=150) -> None:
    st.markdown(_match_summary(team_a, team_b, model, names, actual))
    st.altair_chart(_contribution_chart(team_a, team_b, model, names, height),
                    width="stretch", height=height + 110)
    _contribution_text(team_a, team_b, model, names)
    zero = [p for p in (*team_a, *team_b) if is_guest(p) or normalize_name(p) not in model.ratings]
    st.caption("Each segment is one player: blue adds goals, red costs goals versus an average "
               "player; the black tick is the team total (net). Hover a segment for details."
               + (f" Counted as average (0): {', '.join(zero)}." if zero else ""))


def _player_picker(label: str, key: str) -> list[str]:
    picked = st.multiselect(label, known_players, key=f"{key}_known")
    extra = st.text_input("Other players, comma separated (new players count as average)",
                          key=f"{key}_extra")
    players = list(dict.fromkeys([*picked, *_names(extra)]))
    unknown = [p for p in players if p not in known_players and not is_guest(p)]
    if unknown:
        st.caption(f"No history (counted as average): {', '.join(unknown)}")
    return players


# --- ratings -------------------------------------------------------------------

with tab_ratings:
    if excluded_note:
        st.info(excluded_note)
    unadjusted = models[NO_RATINGS]

    # Change since before the last match day (all matches on the latest date).
    dated = [m.date for m in matches if m.date]
    last_day = max(dated) if dated else None
    before = [m for m in matches if m.date != last_day] if last_day else matches[:-1]
    previous = {v: RatingModel(**options).fit(before, s) for v, s in versions.items()}
    played_before = previous[NO_RATINGS].games

    def _change(v: str, p: str) -> str:
        # Text, not numbers: st.dataframe would show a missing value as "None".
        if unadjusted.games.get(p, 0) and not played_before.get(p, 0):
            return "new"  # first games on the last match day
        d = models[v].rating(p) - previous[v].rating(p)
        return "–" if abs(d) < 0.005 else f"{d:+.2f}"

    table = pd.DataFrame(
        [
            {
                "Player": p,
                **{col: val for v, m in models.items()
                   for col, val in ((v, round(m.rating(p), 2)), (f"{v} Δ", _change(v, p)))},
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

    change_cols = [f"{v} Δ" for v in models]

    def _change_colour(column: pd.Series) -> list[str]:
        return ["color: #1a8a3a" if x.startswith("+") else "color: #c62828" if x.startswith("-")
                else "" for x in column]

    styled = (
        table.style.apply(_highlight, subset=list(user_versions))
        .apply(_change_colour, subset=change_cols)
        .format({v: _signed for v in models} | {"Won": "{:.1f}"})
    )
    st.dataframe(styled, width="stretch", height=min(35 * (len(table) + 1) + 3, 900))
    notes = [f"**{v}**: {len(s)} players rated, 1 point = {models[v].beta:.2f} goals"
             for v, s in user_versions.items()]
    st.caption(
        "Rating = goals a player adds to their team's goal difference per game versus an "
        "average player. Won counts a draw as 0.5. Highlighted cells = you rated that player "
        "in that version. Click a column header to sort."
        + (f"  \nΔ = change from the last match day ({last_day}, "
           f"{sum(m.date == last_day for m in matches)} match(es)); "
           "\"new\" = first games that day." if last_day else "")
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
            _show_contributions(list(s.team_a), list(s.team_b), models[version], height=120)

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
        st.markdown(f"**Player contributions** ({version})")
        _show_contributions(team_a, team_b, models[version])
    else:
        st.info("Pick players for both teams.")

# --- evaluate ------------------------------------------------------------------

with tab_eval:
    if excluded_note:
        st.info(excluded_note)
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


# --- matches: contribution chart (needs the fitted model, so it comes last) ---

with tab_matches:
    st.divider()
    st.markdown(f"**Player contributions in a past match** ({version} ratings)")
    labels = [f"{m.date or '#' + str(i + 1)} · {m.goals_a}-{m.goals_b} · "
              f"{', '.join(m.team_a[:2])}… v {', '.join(m.team_b[:2])}…"
              for i, m in enumerate(all_matches)]
    chosen = st.selectbox("Show match", range(len(all_matches)), index=len(all_matches) - 1,
                          format_func=lambda i: labels[i])
    m = all_matches[chosen]
    _show_contributions(list(m.team_a), list(m.team_b), models[version],
                        names=("Left team", "Right team"), actual=(m.goals_a, m.goals_b))
    st.caption("Ratings come from all included matches, this one too, so past matches look "
               "more predictable than they were. The Evaluate tab shows the honest version."
               + (" This match is excluded from the ratings." if m not in matches else ""))
