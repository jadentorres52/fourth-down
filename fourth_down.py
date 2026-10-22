"""
Fourth-down decision engine: data, training, evaluation, decision logic and API in one file.

Usage
-----
    pip install nflreadpy==0.1.5 polars==2.0.0 pandas==3.0.2 pyarrow==26.0.0 xgboost==3.4.1 scikit-learn==1.8.0 fastapi==0.157.0 uvicorn==0.54.0 joblib==1.5.3
    python fourth_down.py train         # download data, train + evaluate models, save models.joblib,
                                        # and write the models into index.html
    python fourth_down.py export        # only rewrite the models into index.html
    python fourth_down.py check         # re-run the sanity checks against the saved models

Then open index.html in any browser. It runs the models itself, so no server is needed.
    uvicorn fourth_down:app --reload    # optional: a JSON API (and the page) at http://localhost:8000

The big idea
------------
On 4th down a team can go for it, kick a field goal, or punt. For each option we estimate the
chance of each outcome, score the game state that outcome leads to with a win probability (WP)
model, and take the probability-weighted average:

    E[WP | go]   = P(convert) * WP(first down)        + P(fail) * WP(opponent's ball right here)
    E[WP | kick] = P(make)    * WP(up 3, kicking off) + P(miss) * WP(opponent's ball at the kick spot)
    E[WP | punt] =              WP(opponent's ball wherever punts from here usually land)

The recommendation is whichever option is highest. Four models feed those formulas:

    1. Win probability  (XGBoost)    P(team with the ball wins | score, clock, field position, ...)
    2. Conversion       (XGBoost)    P(a 4th-down attempt succeeds | distance, field position, strength)
    3. Field goal       (logistic)   P(the kick is good | distance, indoors)
    4. Punt             (XGBoost)    where the opponent starts after a punt from a given spot

Why not train one model on what coaches actually chose and how it turned out? Coaches don't pick
at random. Historically they went for it mostly in obvious or desperate spots, so that data mixes
up the decision with the situation that prompted it (selection bias). Each component above is a
narrower question ("how often is a 4th-and-2 converted?") that doesn't depend on who chose it.
"""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import json
import re
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field, model_validator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

# ===========================================================================
# Configuration
# ===========================================================================

HERE = Path(__file__).parent
MODEL_PATH = HERE / "models.joblib"        # everything the API needs, in one file
CACHE_PATH = HERE / "pbp_cache.parquet"    # completed seasons, so retraining skips the download
INDEX_PATH = HERE / "index.html"           # the UI: one static file, no npm and no build step

# NFL seasons start in September; January and February games belong to the previous season.
_today = dt.date.today()
CURRENT_SEASON = _today.year if _today.month >= 9 else _today.year - 1

# Win probability and conversion models use complete seasons from 2016 on. By 2016 both the
# longer extra point (2015) and kickoff touchbacks at the 25 (2016) were in place, so late-game
# scoring math looks like today's. More seasons = more data for the harder-to-learn models.
MODEL_SEASONS = list(range(2016, CURRENT_SEASON))

# Field goals and punts use modern kickers and punters only. Long-range field goal accuracy has
# climbed fast (in this data, 50+ yard kicks went from about 59% in 2016 to about 69% in 2024-25),
# and net punting gained close to a yard over the same stretch, so older seasons drag both models
# down. The season in progress is included because recency matters most here.
KICK_SEASONS = list(range(2023, CURRENT_SEASON + 1))

TEST_SEASON = MODEL_SEASONS[-1]   # the last complete season is held out for evaluation
ALL_SEASONS = sorted(set(MODEL_SEASONS) | set(KICK_SEASONS))

# Simplifying assumptions about the states each option leads to. Each one is a fair answer to
# "what would you improve?" in an interview.
PLAY_SECONDS = 6   # game clock a 4th-down play (or kick) burns
TD_POINTS = 7      # a touchdown plus the extra point, which is good almost every time
FG_OFFSET = 18     # kick distance = yards to the end zone + 18 (end zone + snap); median in the data
OPTIONS = ["go", "fg", "punt"]   # this index order is shared with the frontend

# Only these columns are kept from nflverse's ~370-column play-by-play table.
PBP_COLUMNS = [
    "game_id", "play_id", "season", "week", "home_team", "away_team", "posteam", "defteam",
    "qtr", "quarter_seconds_remaining", "game_seconds_remaining", "half_seconds_remaining",
    "down", "ydstogo", "yardline_100", "score_differential", "posteam_timeouts_remaining",
    "defteam_timeouts_remaining", "spread_line", "result", "roof", "play_type",
    "fourth_down_converted", "field_goal_result", "fixed_drive", "drive_start_transition", "desc",
]

# Model inputs, each paired with a monotone constraint: +1 means the prediction may only rise as the
# feature rises, -1 only fall, 0 unconstrained. Tree models otherwise produce small bumps from
# noise, and the UI's sliders expose them instantly (say, conversion odds going *up* when you drag
# from 4th-and-3 to 4th-and-4). The constraints encode football common sense and also regularize.
#
# Every feature is known *before the snap*, which is when the decision gets made. That's why
# there's no "defenders in the box" (recorded at the snap, partly in reaction to the offense) and
# no EPA (computed from the play's result): using them would leak information the coach doesn't
# have when deciding.
WP_FEATURES = {
    "score_differential": 1,          # your score minus theirs
    "diff_time_ratio": 1,             # score difference, amplified as time runs out
    "spread_time": 1,                 # pregame spread, faded as time runs out
    "game_seconds_remaining": 0,      # more time helps the trailing team, hurts the leader
    "half_seconds_remaining": 0,
    "yardline_100": -1,               # yards from the opponent's end zone: farther = worse
    "down": -1,
    "ydstogo": -1,
    "posteam_timeouts_remaining": 1,
    "defteam_timeouts_remaining": -1,
}
CONV_FEATURES = {
    "ydstogo": -1,          # longer = harder
    "yardline_100": 0,      # near the goal line the field compresses, so no fixed direction
    "goal_to_go": 0,        # on 4th-and-goal, converting means scoring a touchdown
    "posteam_spread": 1,    # the pregame line as a stand-in for how good the offense is
}
FG_FEATURES = ["kick_distance", "indoors"]


# ===========================================================================
# Shared feature code
#
# Training and the API both build model inputs through these functions. If the API computed a
# feature even slightly differently than training did ("training/serving skew"), the model would
# return wrong answers without ever raising an error. One code path makes that impossible.
# ===========================================================================

def clock_columns(qtr, quarter_seconds):
    """nflverse's game and half clocks from quarter + clock. Exact in regulation; checked against
    the real columns on every 2025 play. The UI sends quarter + clock because that's how people
    think about time; the models use the game/half clocks."""
    qtr = np.asarray(qtr, dtype=float)
    q = np.asarray(quarter_seconds, dtype=float)
    return q + 900 * (4 - qtr), q + 900 * (qtr % 2 == 1)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derived features from a game state. The two time interactions mirror nflfastR's own win
    probability model: both say that the score matters more, and pregame team strength less,
    as the clock runs down."""
    df = df.copy()
    elapsed = (3600 - df["game_seconds_remaining"]) / 3600
    decay = np.exp(-4 * elapsed)                                 # 1.0 at kickoff, ~0.02 at the end
    df["spread_time"] = df["posteam_spread"] * decay
    df["diff_time_ratio"] = df["score_differential"] / decay
    df["goal_to_go"] = (df["ydstogo"] >= df["yardline_100"]).astype(int)
    df["kick_distance"] = df["yardline_100"] + FG_OFFSET
    df["indoors"] = df["indoors"].astype(int) if "indoors" in df else 0
    return df


def wp_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return add_features(df)[list(WP_FEATURES)]


def conv_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return add_features(df)[list(CONV_FEATURES)]


def fg_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return add_features(df)[FG_FEATURES]


# ===========================================================================
# Data
# ===========================================================================

def download(season: int) -> pd.DataFrame | None:
    """One season of nflverse play-by-play via nflreadpy, trimmed to the columns we use.

    nflreadpy (the official nflverse Python loader, which replaced nfl_data_py in 2025) returns a
    Polars DataFrame. Selecting columns in Polars first keeps memory low, then we convert to pandas
    because scikit-learn and XGBoost's sklearn-style API expect it."""
    import nflreadpy as nfl  # imported here so the API can run without training dependencies

    try:
        print(f"Downloading {season} play-by-play...")
        return nfl.load_pbp(season).select(PBP_COLUMNS).to_pandas()
    except Exception as exc:  # most likely the season hasn't started yet
        print(f"  Skipping {season}: {exc}")
        return None


def load_pbp() -> pd.DataFrame:
    """Completed seasons come from a local cache. The season in progress is always downloaded fresh,
    so retraining once a week picks up the newest games."""
    completed = [s for s in ALL_SEASONS if s < CURRENT_SEASON]
    cached = pd.read_parquet(CACHE_PATH) if CACHE_PATH.exists() else None
    if cached is None or sorted(cached["season"].unique().tolist()) != completed:
        frames = [f for f in (download(s) for s in completed) if f is not None]
        cached = pd.concat(frames, ignore_index=True)
        cached.to_parquet(CACHE_PATH)
    current = download(CURRENT_SEASON) if CURRENT_SEASON in ALL_SEASONS else None
    return pd.concat([cached, current], ignore_index=True) if current is not None else cached


def prepare(pbp: pd.DataFrame) -> pd.DataFrame:
    """Regulation scrimmage plays, plus the columns every model needs."""
    # Plays with a down are scrimmage plays (this drops kickoffs, extra points and timeouts).
    # Overtime is excluded: it has its own clock and possession rules.
    df = pbp[(pbp["qtr"] <= 4) & pbp["down"].notna() & pbp["posteam"].notna()]
    df = df.sort_values(["game_id", "play_id"]).reset_index(drop=True)

    # nflverse reports spread_line and result from the home team's point of view (positive = home
    # favored, home won). Flip both to the point of view of the team with the ball.
    home = (df["posteam"] == df["home_team"]).to_numpy()
    margin = np.where(home, df["result"], -df["result"])
    df = df.assign(
        posteam_spread=np.where(home, df["spread_line"], -df["spread_line"]),
        indoors=df["roof"].isin(["dome", "closed"]).astype(int),
        won=(margin > 0).astype(int),
        tie=margin == 0,
    )

    # For the punt model: who has the ball on the next scrimmage play, and where.
    nxt = df.groupby("game_id")[["posteam", "down", "yardline_100", "qtr"]].shift(-1)
    df["next_posteam"], df["next_down"] = nxt["posteam"], nxt["down"]
    df["next_yardline_100"], df["next_qtr"] = nxt["yardline_100"], nxt["qtr"]
    return df


def wp_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Every scrimmage play, labeled with whether the team that had the ball went on to win.
    Ties are dropped (rare, and they don't fit a win/loss label)."""
    return df[~df["tie"]].dropna(subset=["posteam_spread", "posteam_timeouts_remaining", "result"])


def conversion_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Real 4th-down attempts (runs and passes, including fakes). nflverse's fourth_down_converted
    already counts touchdowns as conversions, which we checked against the 2025 data."""
    att = df[(df["down"] == 4) & df["play_type"].isin(["run", "pass"])]
    return att.dropna(subset=["posteam_spread"])


def field_goal_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Field goal attempts. Blocked kicks count as misses."""
    kicks = df[df["play_type"] == "field_goal"]
    return kicks.assign(made=(kicks["field_goal_result"] == "made").astype(int))


def punt_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Punts, labeled with where the other team started its next play (from its own point of view).
    Reading the next play, instead of adding up kick and return yards, automatically accounts for
    touchbacks, penalties and fair catches. Known gap: punts returned for touchdowns or recovered by
    the kicking team are dropped, which makes punting look slightly safer than it is."""
    p = df[(df["play_type"] == "punt") & (df["next_posteam"] != df["posteam"]) & (df["next_down"] == 1)]
    p = p[(p["next_qtr"] <= 2) == (p["qtr"] <= 2)]  # same half, so halftime doesn't count as a "landing spot"
    return p.assign(opp_yardline_100=p["next_yardline_100"])


# ===========================================================================
# Training
# ===========================================================================

def fit_models(df: pd.DataFrame, model_seasons: list[int], kick_seasons: list[int]) -> dict:
    core = df[df["season"].isin(model_seasons)]

    # 1. Win probability: gradient-boosted trees (XGBoost). Trees handle the interactions that make
    # WP hard (a 7-point lead means little in the 1st quarter and nearly everything with 2 minutes
    # left) without hand-built features for each one. Every play in a game shares one label (who
    # won), so there are only a few thousand truly independent examples despite ~400k rows; shallow
    # trees and big minimum leaf sizes stop the model from memorizing individual games.
    wp_data = wp_rows(core)
    wp = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=200, monotone_constraints=tuple(WP_FEATURES.values()), tree_method="hist",
    )
    wp.fit(wp_matrix(wp_data), wp_data["won"])

    # 2. Conversion: only a few hundred real attempts per season (about 900 in 2025), so a small,
    # heavily constrained model with a handful of features.
    att = conversion_rows(core)
    conv = xgb.XGBClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8, min_child_weight=10,
        monotone_constraints=tuple(CONV_FEATURES.values()), tree_method="hist",
    )
    conv.fit(conv_matrix(att), att["fourth_down_converted"].astype(int))

    # 3. Field goals: logistic regression rather than trees. The problem is basically one feature
    # (distance) with a smooth S-curve, and the model has to extrapolate sensibly to long kicks that
    # are rarely tried. Trees can't extrapolate: past the longest kicks they'd just repeat the last
    # value. Very long kicks get special handling in predict_fg.
    kicks = field_goal_rows(df[df["season"].isin(kick_seasons)])
    fg = LogisticRegression(max_iter=1000).fit(fg_matrix(kicks), kicks["made"])
    fg_max = float(kicks.loc[kicks["made"] == 1, "yardline_100"].max() + FG_OFFSET + 2)

    # 4. Punts: one input (where the punt is from) predicting the average landing spot, trained on
    # recent seasons like field goals. On the 2025 holdout, older punts made the model underrate
    # today's punters by about half a yard; recent seasons only cut that to about 0.1 yards.
    punts = punt_rows(df[df["season"].isin(kick_seasons)])
    punt = xgb.XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, tree_method="hist")
    punt.fit(punts[["yardline_100"]], punts["opp_yardline_100"])

    # After any score, the other team gets a kickoff. Kickoff rules changed in 2024 and 2025, so
    # measure where drives start after kickoffs in the most recent season instead of hard-coding it.
    latest = df[df["season"] == max(model_seasons + kick_seasons)]
    firsts = latest.groupby(["game_id", "fixed_drive"]).head(1)
    kickoff_start = float(firsts.loc[firsts["drive_start_transition"] == "KICKOFF", "yardline_100"].mean())

    return {"wp": wp, "conv": conv, "fg": fg, "fg_max": fg_max, "punt": punt, "kickoff_start": kickoff_start}


def calibration(name: str, y, p, baseline: float | None = None) -> dict:
    """Print how well predicted probabilities match reality, bucketed by prediction.

    Calibration matters more than accuracy here: the probabilities get multiplied into expected
    win probabilities, so a model that says 60% when the truth is 45% gives confidently wrong
    advice. Brier score is mean squared error on probabilities (lower is better); calibration
    error is the average gap between predicted and actual rates across the buckets."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    buckets = np.clip((p * 10).astype(int), 0, 9)
    ece = sum(abs(p[buckets == k].mean() - y[buckets == k].mean()) * (buckets == k).mean()
              for k in range(10) if (buckets == k).any())
    line = f"{name}: Brier {brier_score_loss(y, p):.4f}, log loss {log_loss(y, p):.4f}, calibration error {ece:.4f}"
    if baseline is not None:  # the score you'd get by always guessing the training average
        line += f" (always guessing the average: Brier {brier_score_loss(y, np.full_like(p, baseline)):.4f})"
    print(line)
    for k in range(10):
        m = buckets == k
        if m.sum() >= 25:
            print(f"    predicted {p[m].mean():.2f}  actual {y[m].mean():.2f}  n={m.sum()}")
    return {"brier": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, p)), "ece": float(ece)}


def evaluate_holdout(df: pd.DataFrame) -> dict:
    """Train on earlier seasons, score on TEST_SEASON.

    Splitting by season instead of by random plays matters twice over: plays from one game are
    strongly correlated (same teams, same final result), so a random split puts near-duplicates in
    both train and test and inflates the scores; and testing on a later season is how the model
    actually gets used."""
    m = fit_models(df, [s for s in MODEL_SEASONS if s < TEST_SEASON], [s for s in KICK_SEASONS if s < TEST_SEASON])
    train = df[df["season"] < TEST_SEASON]
    test = df[df["season"] == TEST_SEASON]
    print(f"\nHoldout evaluation on the {TEST_SEASON} season (models trained only on earlier seasons)\n")
    metrics = {}

    w = wp_rows(test)
    metrics["wp"] = calibration("Win probability", w["won"], m["wp"].predict_proba(wp_matrix(w))[:, 1])

    a_tr, a_te = conversion_rows(train), conversion_rows(test)
    metrics["conv"] = calibration("4th-down conversion", a_te["fourth_down_converted"],
                                  m["conv"].predict_proba(conv_matrix(a_te))[:, 1],
                                  a_tr["fourth_down_converted"].mean())

    k_tr = field_goal_rows(train[train["season"].isin(KICK_SEASONS)])
    k_te = field_goal_rows(test)
    metrics["fg"] = calibration("Field goal", k_te["made"], predict_fg(m, k_te), k_tr["made"].mean())

    p_tr, p_te = punt_rows(train[train["season"].isin(KICK_SEASONS)]), punt_rows(test)
    mae = np.abs(m["punt"].predict(p_te[["yardline_100"]]) - p_te["opp_yardline_100"]).mean()
    base = np.abs(p_tr["opp_yardline_100"].mean() - p_te["opp_yardline_100"]).mean()
    print(f"Punt landing spot: average miss {mae:.1f} yards (always guessing the average: {base:.1f})")
    metrics["punt_mae"] = float(mae)
    return metrics


def real_fourth_downs(df: pd.DataFrame) -> list[dict]:
    """Every real 4th-down decision from the most recent season, for the "model vs. coach" feature.
    When that's the season in progress, the win probability and conversion models never trained on
    these plays, so the comparison is honest."""
    latest = df[(df["season"] == df["season"].max()) & (df["down"] == 4)]
    plays = latest[latest["play_type"].isin(["run", "pass", "punt", "field_goal"]) & (latest["ydstogo"] >= 1)]
    plays = plays.dropna(subset=["posteam_spread", "posteam_timeouts_remaining", "quarter_seconds_remaining"])
    choice = plays["play_type"].map({"run": "go", "pass": "go", "punt": "punt", "field_goal": "fg"})
    out = pd.DataFrame({
        "yardline_100": plays["yardline_100"].astype(int),
        "ydstogo": plays["ydstogo"].astype(int),
        "score_differential": plays["score_differential"].astype(int),
        "qtr": plays["qtr"].astype(int),
        "quarter_seconds_remaining": plays["quarter_seconds_remaining"].astype(int),
        "posteam_timeouts_remaining": plays["posteam_timeouts_remaining"].astype(int),
        "defteam_timeouts_remaining": plays["defteam_timeouts_remaining"].astype(int),
        "posteam_spread": plays["posteam_spread"].astype(float),
        "indoors": plays["indoors"].astype(bool),
        "choice": choice, "desc": plays["desc"], "posteam": plays["posteam"], "defteam": plays["defteam"],
        "season": plays["season"].astype(int), "week": plays["week"].astype(int),
    })
    return out.to_dict("records")


def train() -> None:
    df = prepare(load_pbp())
    metrics = evaluate_holdout(df)

    # Once the holdout has told us how good the approach is, refit on everything so the saved
    # models include the most recent data.
    print(f"\nRefitting: WP/conversion on {MODEL_SEASONS[0]}-{MODEL_SEASONS[-1]}, "
          f"field goals and punts on {KICK_SEASONS[0]}-{KICK_SEASONS[-1]}...")
    bundle = fit_models(df, MODEL_SEASONS, KICK_SEASONS)
    bundle.update(metrics=metrics, model_seasons=MODEL_SEASONS, kick_seasons=KICK_SEASONS,
                  test_season=TEST_SEASON, plays=real_fourth_downs(df))
    joblib.dump(bundle, MODEL_PATH)
    print(f"Saved {MODEL_PATH.name}. After kickoffs, drives start {100 - bundle['kickoff_start']:.0f} "
          f"yards from their own goal line on average. Kicks past {bundle['fg_max']:.0f} yards are treated as 1% shots.\n")
    check(bundle)
    export_to_page(bundle)


# ===========================================================================
# Decision engine
#
# Vectorized: each row of the input is one situation, and every model is called once per batch.
# That's what lets /api/grid score ~1,400 situations for the heatmap in a single request.
# ===========================================================================

def predict_fg(b: dict, states: pd.DataFrame) -> np.ndarray:
    """P(the kick is good). More than 2 yards past the longest make in the training data, the
    logistic curve is replaced by a flat 1%: low enough that the engine never seriously considers
    the kick, but not a hard zero. The 2025 holdout showed why: the earlier seasons' longest make
    was 65 yards, then a 68-yarder went in. A 0% prediction on a make is penalized without limit
    by log loss (it jumped from 0.36 to 0.39 on that single kick)."""
    p = b["fg"].predict_proba(fg_matrix(states))[:, 1]
    return np.where(add_features(states)["kick_distance"] > b["fg_max"], np.minimum(p, 0.01), p)


def evaluate(b: dict, s: pd.DataFrame) -> pd.DataFrame:
    """Expected win probability of each option for each situation (row) in s."""
    s = s.reset_index(drop=True)
    y = s["yardline_100"].to_numpy(float)
    togo = s["ydstogo"].to_numpy(float)
    diff = s["score_differential"].to_numpy(float)
    ours = s["posteam_timeouts_remaining"].to_numpy(float)
    theirs = s["defteam_timeouts_remaining"].to_numpy(float)
    spread = s["posteam_spread"].to_numpy(float)
    gsr, half = s["game_seconds_remaining"].to_numpy(float), s["half_seconds_remaining"].to_numpy(float)

    # Clock after the play. Taking the time off the half clock first means a play with 3 seconds
    # left in the 2nd quarter ends the half instead of spilling into the 3rd.
    half_after = np.maximum(half - PLAY_SECONDS, 0)
    gsr_after = gsr - (half - half_after)

    def wp(yl, score, down, dist, offense_is_us: bool, after_play: bool = True) -> np.ndarray:
        """Win probability for whichever team has the ball in a hypothetical state.

        This is why we train our own WP model: the play-by-play data only has win probability for
        situations that actually happened, but here we need it for ones that didn't (like "what if
        they had punted instead?"). When the other team has the ball, everything is flipped to
        their point of view: their score margin, their timeouts, their side of the spread."""
        state = pd.DataFrame({
            "yardline_100": yl, "ydstogo": dist, "down": down, "score_differential": score,
            "game_seconds_remaining": gsr_after if after_play else gsr,
            "half_seconds_remaining": half_after if after_play else half,
            "posteam_timeouts_remaining": ours if offense_is_us else theirs,
            "defteam_timeouts_remaining": theirs if offense_is_us else ours,
            "posteam_spread": spread if offense_is_us else -spread,
        })
        return b["wp"].predict_proba(wp_matrix(state))[:, 1]

    def our_first_down(yl, score):
        return wp(yl, score, 1, np.minimum(10, yl), True)

    def their_first_down(yl, our_score):
        yl = np.clip(yl, 1, 99)
        return 1 - wp(yl, -our_score, 1, np.minimum(10, yl), False)  # our WP = 1 - theirs

    # Go for it. Success: first down at the line to gain (a touchdown, then a kickoff, if it's goal
    # to go). Placing the ball exactly at the sticks undersells going for it a little, because real
    # conversions often gain extra yards. Failure: the opponent takes over at the line of scrimmage.
    p_conv = b["conv"].predict_proba(conv_matrix(s))[:, 1]
    goal = togo >= y
    wp_conv = np.where(goal, their_first_down(np.full_like(y, b["kickoff_start"]), diff + TD_POINTS),
                       our_first_down(np.maximum(y - togo, 1), diff))
    wp_fail = their_first_down(100 - y, diff)
    go = p_conv * wp_conv + (1 - p_conv) * wp_fail

    # Field goal. Make: up 3 more, then kick off. Miss: NFL rules give the opponent the ball at the
    # spot of the kick (8 yards behind the line), or at their 20 if that spot is inside the 20.
    p_make = predict_fg(b, s)
    wp_make = their_first_down(np.full_like(y, b["kickoff_start"]), diff + 3)
    kick_spot = y + FG_OFFSET - 10
    wp_miss = their_first_down(np.minimum(100 - kick_spot, 80), diff)
    fg = p_make * wp_make + (1 - p_make) * wp_miss

    # Punt: the opponent starts wherever punts from this spot usually leave them.
    opp_start = np.clip(b["punt"].predict(pd.DataFrame({"yardline_100": y})), 1, 99)
    punt = their_first_down(opp_start, diff)

    out = pd.DataFrame({
        "go": go, "fg": fg, "punt": punt,
        "wp_now": wp(y, diff, 4, togo, True, after_play=False),   # before the decision, for context
        "p_convert": p_conv, "wp_convert": wp_conv, "wp_fail": wp_fail,
        "p_make": p_make, "kick_distance": y + FG_OFFSET, "wp_make": wp_make, "wp_miss": wp_miss,
        "punt_opp_yardline_100": opp_start,
    })
    vals = out[OPTIONS].to_numpy()
    order = np.argsort(-vals, axis=1)
    rows = np.arange(len(out))
    out["best"] = np.array(OPTIONS)[order[:, 0]]
    out["margin"] = vals[rows, order[:, 0]] - vals[rows, order[:, 1]]   # how much better than the runner-up

    # Break-even: the conversion rate at which going for it exactly ties the better of kicking and
    # punting. Solve p * wp_conv + (1 - p) * wp_fail = best_alternative for p. This turns "the
    # model says go" into something checkable: "you need 31%, and teams convert 55% of these".
    with np.errstate(divide="ignore", invalid="ignore"):
        breakeven = (np.maximum(fg, punt) - wp_fail) / (wp_conv - wp_fail)
    out["breakeven"] = np.where(wp_conv > wp_fail, np.clip(breakeven, 0, 1), np.nan)
    return out


def states_frame(rows: list[dict]) -> pd.DataFrame:
    """API-shaped situations (quarter + clock) to the clock columns the models use."""
    s = pd.DataFrame(rows)
    s["game_seconds_remaining"], s["half_seconds_remaining"] = clock_columns(s["qtr"], s["quarter_seconds_remaining"])
    return s


# ===========================================================================
# Sanity checks: situations with an obvious right answer, plus the slider-style monotonicity
# guarantees. Runs after every training, like unit tests for model behavior.
# ===========================================================================

SANITY_CASES = [
    ("4th & 3 at the opp 42, down 4, 2:00 left: a field goal doesn't even tie it",
     dict(yardline_100=42, ydstogo=3, score_differential=-4, qtr=4, quarter_seconds_remaining=120), "go"),
    ("4th & 15 at your own 20, tied, 1st quarter",
     dict(yardline_100=80, ydstogo=15, score_differential=0, qtr=1, quarter_seconds_remaining=600), "punt"),
    ("4th & 8 at the opp 15, tied, 2nd quarter",
     dict(yardline_100=15, ydstogo=8, score_differential=0, qtr=2, quarter_seconds_remaining=600), "fg"),
    ("4th & 2 at the opp 20, down 2, 0:05 left: kick the game-winner",
     dict(yardline_100=20, ydstogo=2, score_differential=-2, qtr=4, quarter_seconds_remaining=5), "fg"),
    ("4th & goal at the 1, tied, 1st quarter",
     dict(yardline_100=1, ydstogo=1, score_differential=0, qtr=1, quarter_seconds_remaining=600), "go"),
]
DEFAULTS = dict(posteam_timeouts_remaining=3, defteam_timeouts_remaining=3, posteam_spread=0.0, indoors=False)


def check(b: dict | None = None) -> bool:
    b = b or joblib.load(MODEL_PATH)
    ok = True
    res = evaluate(b, states_frame([{**DEFAULTS, **case} for _, case, _ in SANITY_CASES]))
    for (label, _, want), got in zip(SANITY_CASES, res["best"]):
        ok &= got == want
        print(f"{'PASS' if got == want else 'FAIL'}  {label} -> {got} (expected {want})")

    base = {**DEFAULTS, **SANITY_CASES[0][1]}
    ladder = evaluate(b, states_frame([{**base, "ydstogo": d} for d in range(1, 16)]))
    kicks = evaluate(b, states_frame([{**base, "yardline_100": yl, "ydstogo": 1} for yl in range(1, 60)]))
    conv_ok = bool(np.all(np.diff(ladder["p_convert"]) <= 1e-9))
    fg_ok = bool(np.all(np.diff(kicks["p_make"]) <= 1e-9))
    print(f"{'PASS' if conv_ok else 'FAIL'}  conversion odds never rise as yards to go rises")
    print(f"{'PASS' if fg_ok else 'FAIL'}  field goal odds never rise as the kick gets longer")
    return ok and conv_ok and fg_ok


# ===========================================================================
# Export for the browser
#
# index.html runs the models itself, so the page works with no server: opened from a file, on a
# static host, or on a phone. This writes the models into the page as plain numbers (each XGBoost
# tree as four arrays, plus the logistic regression's coefficients), and checks every export
# against the real models before writing it. The page's JavaScript mirrors evaluate() above.
# ===========================================================================

def _browser_trees(model, probe: pd.DataFrame) -> dict:
    """An XGBoost model as plain arrays: per tree [split feature, split value, left child, right child].
    At a leaf (left child -1) the split value slot holds the leaf's output instead."""
    raw = json.loads(model.get_booster().save_raw(raw_format="json"))
    trees = [[t["split_indices"], [float(f"{v:.9g}") for v in t["split_conditions"]], t["left_children"], t["right_children"]]
             for t in raw["learner"]["gradient_booster"]["model"]["trees"]]

    def tree_sum(row):  # the same loop the page runs, including XGBoost's 32-bit float comparisons
        total = 0.0
        for feature, value, left, right in trees:
            i = 0
            while left[i] != -1:
                i = left[i] if np.float32(row[feature[i]]) < np.float32(value[i]) else right[i]
            total += value[i]
        return total

    sums = np.array([tree_sum(row) for row in probe.to_numpy(dtype=float)])
    margins = model.get_booster().predict(xgb.DMatrix(probe), output_margin=True)
    base = float(np.mean(margins - sums))  # the constant every prediction starts from
    worst = float(np.max(np.abs(margins - sums - base)))
    if worst > 1e-4:
        raise RuntimeError(f"Exported trees don't reproduce the model (off by up to {worst:.2g})")
    return {"features": list(probe.columns), "base": base, "trees": trees}


def browser_models(b: dict) -> dict:
    rng = np.random.default_rng(0)  # random but realistic situations to check the export against
    n = 300
    yl = rng.integers(1, 100, n)
    gsr, half = clock_columns(rng.integers(1, 5, n), rng.integers(0, 901, n))
    probe = pd.DataFrame({
        "yardline_100": yl, "ydstogo": np.minimum(rng.integers(1, 16, n), yl), "down": rng.integers(1, 5, n),
        "score_differential": rng.integers(-21, 22, n), "game_seconds_remaining": gsr, "half_seconds_remaining": half,
        "posteam_timeouts_remaining": rng.integers(0, 4, n), "defteam_timeouts_remaining": rng.integers(0, 4, n),
        "posteam_spread": rng.choice([-7.0, -3.0, 0.0, 3.0, 7.0], n), "indoors": 0,
    })
    return {
        "wp": _browser_trees(b["wp"], wp_matrix(probe)),
        "conv": _browser_trees(b["conv"], conv_matrix(probe)),
        "punt": _browser_trees(b["punt"], probe[["yardline_100"]]),
        "fg": {"intercept": float(b["fg"].intercept_[0]), "coef": [float(c) for c in b["fg"].coef_[0]], "max": b["fg_max"]},
        "kickoff_start": b["kickoff_start"],
        "constants": {"play_seconds": PLAY_SECONDS, "td_points": TD_POINTS, "fg_offset": FG_OFFSET},
        "seasons": {"model": b["model_seasons"], "kick": b["kick_seasons"]},
    }


def export_to_page(b: dict | None = None) -> bool:
    """Write the models into index.html, inside <script type="application/json" id="models">.

    If the page is missing, or is an older copy without that block, this explains what to do
    instead of crashing. Training has already saved models.joblib by the time this runs, so
    fixing the page only takes `python fourth_down.py export`, not a full retrain."""
    fix = "Put the latest index.html (about 250 KB) next to fourth_down.py, then run: python fourth_down.py export"
    if not INDEX_PATH.exists():
        print(f"Models saved, but not written into the page: there's no {INDEX_PATH.name} here. {fix}")
        return False
    page = INDEX_PATH.read_text(encoding="utf-8")
    # Finds the block even if an editor reordered the tag's attributes or changed its quotes.
    block = re.search(r"""<script\b[^>]*\bid=["']models["'][^>]*>(.*?)</script>""", page, flags=re.S)
    if not block:
        print(f"Models saved, but not written into the page: this {INDEX_PATH.name} has no models block, "
              f"so it's an older copy of the page. {fix}")
        return False
    b = b or joblib.load(MODEL_PATH)
    payload = json.dumps(browser_models(b), separators=(",", ":"))
    INDEX_PATH.write_text(page[:block.start(1)] + payload + page[block.end(1):], encoding="utf-8")
    print(f"Wrote the models into {INDEX_PATH.name} ({len(payload) // 1024} KB). Open it in a browser; no server needed.")
    return True


# ===========================================================================
# API (FastAPI)
#
# Request fields use nflverse's column names, so a request looks exactly like a row of the
# training data and nothing gets renamed between the frontend, the API and the models.
# Pydantic validates every request before any model code runs.
# ===========================================================================

class Context(BaseModel):
    """Everything about a situation except where the ball is."""
    score_differential: int = Field(ge=-50, le=50, description="Your score minus theirs")
    qtr: int = Field(ge=1, le=4)
    quarter_seconds_remaining: int = Field(ge=0, le=900)
    posteam_timeouts_remaining: int = Field(3, ge=0, le=3)
    defteam_timeouts_remaining: int = Field(3, ge=0, le=3)
    posteam_spread: float = Field(0.0, ge=-25, le=25, description="Pregame line; positive = you're favored")
    indoors: bool = False


class GameState(Context):
    yardline_100: int = Field(ge=1, le=99, description="Yards from the opponent's end zone")
    ydstogo: int = Field(ge=1, le=99)

    @model_validator(mode="after")
    def distance_fits(self):
        # 4th-and-10 from the opponent's 5 can't happen: it would be 4th-and-goal from the 5.
        if self.ydstogo > self.yardline_100:
            raise ValueError("ydstogo can't be longer than the distance to the end zone")
        return self


MODELS: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load the models once at startup (not per request), and grade every real play up front so
    "show me a play the model disagrees with" is just a filter."""
    if MODEL_PATH.exists():
        b = joblib.load(MODEL_PATH)
        plays = pd.DataFrame(b["plays"])
        res = evaluate(b, states_frame(plays.to_dict("records")))
        chosen = res[OPTIONS].to_numpy()[np.arange(len(res)), plays["choice"].map(OPTIONS.index).to_numpy()]
        plays["wp_cost"] = res[OPTIONS].max(axis=1).to_numpy() - chosen
        plays["model_choice"] = res["best"].to_numpy()
        b["plays_df"] = plays
        MODELS.update(b)
    yield


# FastAPI's built-in /docs and /redoc pages load Swagger UI and ReDoc from a CDN by floating major
# version with no integrity hash, so whatever gets published next would run in your browser.
# They're turned off; the raw API schema is still available at /openapi.json.
app = FastAPI(title="Fourth-down decision engine", lifespan=lifespan, docs_url=None, redoc_url=None)


def models() -> dict:
    if not MODELS:
        raise HTTPException(503, "No trained models found. Run: python fourth_down.py train")
    return MODELS


def clean(x: float):
    """JSON has no NaN, so missing numbers go out as null."""
    return None if np.isnan(x) else x


@app.get("/api/health")
def health():
    b = models()
    return {"model_seasons": b["model_seasons"], "kick_seasons": b["kick_seasons"],
            "test_season": b["test_season"], "metrics": b["metrics"]}


@app.post("/api/evaluate")
def evaluate_one(state: GameState):
    """Expected win probability of each option for one situation, plus the pieces behind it."""
    r = evaluate(models(), states_frame([state.model_dump()])).iloc[0]
    f = lambda k: clean(float(r[k]))  # noqa: E731
    return {
        "best": r["best"], "margin": f("margin"), "wp_now": f("wp_now"), "breakeven": f("breakeven"),
        "options": {
            "go": {"wp": f("go"), "p_convert": f("p_convert"), "wp_convert": f("wp_convert"), "wp_fail": f("wp_fail")},
            "fg": {"wp": f("fg"), "p_make": f("p_make"), "kick_distance": f("kick_distance"),
                   "wp_make": f("wp_make"), "wp_miss": f("wp_miss")},
            "punt": {"wp": f("punt"), "opp_yardline_100": f("punt_opp_yardline_100")},
        },
    }


@app.post("/api/grid")
def decision_grid(ctx: Context, max_ydstogo: int = 15):
    """The recommended call at every field position and distance for one score and clock.
    Cells are compact [yardline_100, ydstogo, option index, margin] lists to keep the payload small."""
    rows = [{**ctx.model_dump(), "yardline_100": yl, "ydstogo": d}
            for yl in range(1, 100) for d in range(1, min(max_ydstogo, yl) + 1)]
    res = evaluate(models(), states_frame(rows))
    return {
        "max_ydstogo": max_ydstogo,
        "cells": [[row["yardline_100"], row["ydstogo"], OPTIONS.index(best), round(float(m), 4)]
                  for row, best, m in zip(rows, res["best"], res["margin"])],
    }


@app.get("/api/plays/random")
def random_play(disagree: bool = False):
    """A real 4th down from the most recent season. With disagree=true, only plays where the model
    would have chosen differently and the coach's call cost at least 2 points of win probability."""
    plays = models()["plays_df"]
    if disagree:
        plays = plays[(plays["model_choice"] != plays["choice"]) & (plays["wp_cost"] >= 0.02)]
    if plays.empty:
        raise HTTPException(404, "No plays match that filter yet.")
    p = plays.sample(1).iloc[0]
    state = {k: p[k].item() if hasattr(p[k], "item") else p[k] for k in GameState.model_fields}
    return {"state": state, "choice": p["choice"], "desc": p["desc"], "posteam": p["posteam"],
            "defteam": p["defteam"], "season": int(p["season"]), "week": int(p["week"])}


# ===========================================================================
# The UI: one static HTML file, a React app with no npm and no build step (React and htm load
# from a CDN, pinned to exact versions and checked with integrity hashes), served by this same server. Page and API share one origin, so no CORS setup is needed (and
# none is enabled: other websites' scripts can't call this API from a visitor's browser).
# ===========================================================================

def content_security_policy(html: bytes) -> str:
    """A Content-Security-Policy that pins the page's inline <script> and <style> by hash.

    The browser will run only the page's own inline script (pinned by hash) and the exact library
    files the page names by URL (each also verified by its integrity attribute): nothing injected,
    nothing else from other sites, no eval. The page can fetch only from this server, and the only
    other outside resource it may load is Google Fonts. The hashes are computed from the file as it's served,
    so editing index.html never breaks the policy. Line endings are normalized first because a
    browser's HTML parser turns CRLF into LF before hashing (matters if the file is edited on Windows)."""
    def hashes(tag: bytes) -> str:
        blocks = re.findall(rb"<" + tag + rb"\b[^>]*>(.*?)</" + tag + rb">", html, flags=re.S)
        digests = (hashlib.sha256(b.replace(b"\r\n", b"\n").replace(b"\r", b"\n")).digest() for b in blocks)
        return " ".join(f"'sha256-{base64.b64encode(d).decode()}'" for d in digests)

    # Library files the page loads by URL, allowed by their exact address.
    external = " ".join(u.decode() for u in re.findall(rb'<script\b[^>]*\bsrc="(https://[^"]+)"', html))
    return "; ".join([
        "default-src 'none'",
        f"script-src {hashes(b'script')} {external}",
        f"style-src {hashes(b'style')} https://fonts.googleapis.com",
        "font-src https://fonts.gstatic.com",
        "img-src 'self' data:",
        "connect-src 'self'",
        "base-uri 'none'",
        "form-action 'none'",
        "frame-ancestors 'none'",
    ])


@app.get("/", include_in_schema=False)
def index():
    """The page, read fresh on each request so edits show up on refresh without a restart."""
    if not INDEX_PATH.exists():
        raise HTTPException(404, "index.html wasn't found next to fourth_down.py")
    html = INDEX_PATH.read_bytes()
    return Response(html, media_type="text/html; charset=utf-8", headers={
        "Content-Security-Policy": content_security_policy(html),
        "Referrer-Policy": "no-referrer",       # don't leak page URLs to the font server
        "X-Content-Type-Options": "nosniff",
    })


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "train":
        train()
    elif command == "export":
        sys.exit(0 if export_to_page() else 1)
    elif command == "check":
        sys.exit(0 if check() else 1)
    else:
        print(__doc__)
