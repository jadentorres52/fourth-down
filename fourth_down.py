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

# Win probability and conversion models use complete seasons from 2011 on, when kickoffs moved up
# to the 35 and the modern touchback era began. Starting in 2016 instead (closer to today's rules)
# left the win probability model too little data to learn what field position is worth early in
# games: it called 4th & 3 at your own 11 in the 1st quarter a go. Adding 2011-2015 improved its
# log loss on the 2025 holdout from 0.4823 to 0.4790 and turned that call into a punt.
MODEL_SEASONS = list(range(2011, CURRENT_SEASON))

# Field goals and punts use modern kickers and punters only. Long-range field goal accuracy has
# climbed fast (in this data, 50+ yard kicks went from about 59% in 2016 to about 69% in 2024-25),
# and net punting gained close to a yard over the same stretch, so older seasons drag both models
# down. The season in progress is included because recency matters most here.
KICK_SEASONS = list(range(2023, CURRENT_SEASON + 1))

# The win probability model uses every complete season nflverse has, back to 1999. It has the most to
# learn (every play, and only one result per game) and was the least stable on less data.
WP_SEASONS = list(range(1999, CURRENT_SEASON))

TEST_SEASON = MODEL_SEASONS[-1]   # the last complete season is held out for evaluation
ALL_SEASONS = sorted(set(MODEL_SEASONS) | set(KICK_SEASONS) | set(WP_SEASONS))

# Simplifying assumptions about the states each option leads to. Each one is a fair answer to
# "what would you improve?" in an interview.
PLAY_SECONDS = 6   # game clock a 4th-down play (or kick) burns
TD_POINTS = 7      # a touchdown plus the extra point, which is good almost every time
FG_OFFSET = 18     # kick distance = yards to the end zone + 18 (end zone + snap); median in the data
OPTIONS = ["go", "fg", "punt"]   # this index order is shared with the frontend

# Tree models are step functions, so a learned curve can jump between neighboring yard lines. When two
# options are close, those jumps made the call flip back and forth from one yard line to the next.
# The engine averages each learned field-position effect over the two yard lines on each side,
# weighted toward the middle. Pairs are (yard offset, weight); the page reads the same list.
SMOOTHING = [(-2, 1), (-1, 2), (0, 3), (1, 2), (2, 1)]

# A deliberate conservative lean. This is a product choice, not something the models estimate: early in a
# game, failing on 4th down in your own half counts extra against going for it. The penalty is
# FAIL_PENALTY of win probability at your own 1, shrinking to nothing at your own 45 (OWN_HALF_LINE), and
# it fades out over the 3rd quarter so late-game comebacks are untouched. Pure win probability calls
# many own-territory spots near coin flips (4th & 2 at your own 16 in a tied 1st quarter: 44.7% to go
# for it vs 44.4% to punt), and few coaches would risk handing over a short field that early.
FAIL_PENALTY = 0.15
OWN_HALF_LINE = 55   # yardline_100 of your own 45

# A second lean, also a product choice: early in a game, take the points. The raw models go for it on
# 4th & 5 everywhere inside the opponent's 35, and treat a 59-yard try from the opponent's 41 as a coin
# flip against punting. For kicks of 59 yards or shorter (the opponent's 41 and in), the field goal
# gets a bonus of up to FG_BONUS of win probability, times the chance the kick is good. It phases in
# over the FG_RAMP yards before the 41 and fades out over the 3rd quarter like the penalty above.
FG_BONUS = 0.03
FG_RANGE_LINE = 44   # yardline_100 where the bonus starts (a 62-yard try)
FG_RAMP = 3          # yards to reach full strength, at the opponent's 41

# 4th-and-inches is logged as 4th & 1 in the play-by-play, so the data can't say how much easier it is.
# The calculator assumes inches cuts the chance of failing by a third, another judgment call: at
# midfield, a 68% conversion chance for a full yard becomes 79% for inches. ydstogo = 0 means inches.
INCHES_FAIL_SHARE = 2 / 3

# When the top two options are within TOSS_UP of each other, the calculator says it's a toss-up instead of
# making a call, because the models can't reliably separate them. The exception is where one of the leans
# is in play (before the 4th quarter, in your own half or in field goal range): there the lean decides.
TOSS_UP = 0.01

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
# Win probability: my own model, built the way nflverse builds theirs. An earlier version of mine fit its
# training games far better than unseen ones (log loss 0.441 vs 0.479) and its calls in rare late-game spots
# flipped when retrained on resampled history. This one copies what works in nflverse's published model:
# its inputs, every season back to 1999, small trees (theirs are at most 4 deep), monotone constraints,
# and a tree count picked by early stopping on a held-out season. It scores the same as theirs on seasons
# neither has trained on, with no gap between training and unseen data. "home" and "receive_2h_ko" (does
# the offense get the 2nd-half kickoff) aren't calculator inputs, so evaluate() averages over both.
WP_FEATURES = ["receive_2h_ko", "home", "half_seconds_remaining", "game_seconds_remaining", "Diff_Time_Ratio",
               "score_differential", "down", "ydstogo", "yardline_100", "posteam_timeouts_remaining",
               "defteam_timeouts_remaining"]
# +1: win probability can only rise as the input rises, -1 only fall, 0 free (same order as WP_FEATURES).
WP_PARAMS = dict(objective="binary:logistic", eta=0.1, max_depth=5, min_child_weight=100, subsample=0.8, colsample_bytree=0.8,
                 monotone_constraints="(1,1,0,0,1,1,-1,-1,-1,1,-1)", tree_method="hist")
WP_TREES = 280   # where log loss on the held-out 2023 season stopped improving
# nflverse's published model, downloaded during training only as a benchmark to compare against.
NFLVERSE_WP_URL = "https://raw.githubusercontent.com/nflverse/fastrmodels/master/data/wp_model.rda"
NFLVERSE_WP_PATH = HERE / "nflverse_wp_model.ubj"
CONV_FEATURES = {
    "ydstogo": -1,          # longer = harder
    "yardline_capped": 1,   # field position, but only inside the 20, where the field compresses and
                            # converting gets harder. Farther out it's held constant: left free, the
                            # model learned from rare, unusual attempts deep in a team's own territory
                            # and gave 4th & 3 at your own 11 a 63% chance (the league converts ~51%).
    "goal_to_go": 0,        # on 4th-and-goal, converting means scoring a touchdown
    "posteam_spread": 1,    # the pregame line as a stand-in for how good the offense is
}
FG_FEATURES = ["kick_distance", "indoors"]
STATE_COLUMNS = ["yardline_100", "ydstogo", "down", "score_differential", "game_seconds_remaining", "half_seconds_remaining",
                 "posteam_timeouts_remaining", "defteam_timeouts_remaining", "posteam_spread", "indoors"]
RED_ZONE = 20   # yardline_capped = min(yards to the end zone, RED_ZONE)


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
    df["yardline_capped"] = np.minimum(df["yardline_100"], RED_ZONE)
    df["kick_distance"] = df["yardline_100"] + FG_OFFSET
    df["indoors"] = df["indoors"].astype(int) if "indoors" in df else 0
    return df


def wp_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Inputs for nflverse's win probability model. Expects "home" and "receive_2h_ko" columns."""
    f = df.copy()
    f["Diff_Time_Ratio"] = f["score_differential"] / np.exp(-4 * (3600 - f["game_seconds_remaining"]) / 3600)
    f["receive_2h_ko"] = np.where(f["half_seconds_remaining"] < f["game_seconds_remaining"], f["receive_2h_ko"], 0)  # 2nd half: always 0
    return f[WP_FEATURES].astype("float32")


def wp_predict(model, df: pd.DataFrame) -> np.ndarray:
    return model.predict(xgb.DMatrix(wp_matrix(df).to_numpy()))


def load_nflverse_wp() -> xgb.Booster:
    """nflverse's published win probability model. The R package stores it as a raw vector inside an
    .rda file; this pulls the XGBoost bytes out of R's serialization format and caches them."""
    if not NFLVERSE_WP_PATH.exists():
        import bz2, gzip, lzma, struct, urllib.request
        data = urllib.request.urlopen(NFLVERSE_WP_URL).read()
        for decompress in (lzma.decompress, gzip.decompress, bz2.decompress):
            try:
                data = decompress(data); break
            except Exception:
                pass
        for i in range(len(data) - 8):  # a raw vector: type 24 in a 4-byte header, then its length, then the bytes
            if data[i:i + 3] == b"\x00\x00\x00" and data[i + 3] == 24:
                n = struct.unpack(">i", data[i + 4:i + 8])[0]
                if 1000 < n <= len(data) - i - 8 and data[i + 8:i + 9] == b"{":
                    NFLVERSE_WP_PATH.write_bytes(data[i + 8:i + 8 + n]); break
    model = xgb.Booster()
    model.load_model(bytearray(NFLVERSE_WP_PATH.read_bytes()))
    return model


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
        return _trim(nfl.load_pbp(season).select(PBP_COLUMNS).to_pandas())
    except Exception as exc:  # most likely the season hasn't started yet
        print(f"  Skipping {season}: {exc}")
        return None


def load_pbp() -> pd.DataFrame:
    """Completed seasons come from a local cache. The season in progress is always downloaded fresh,
    so retraining once a week picks up the newest games."""
    completed = [s for s in ALL_SEASONS if s < CURRENT_SEASON]
    cached = pd.read_parquet(CACHE_PATH) if CACHE_PATH.exists() else None
    if cached is None or sorted(cached["season"].unique().tolist()) != completed or not set(PBP_COLUMNS) <= set(cached.columns):
        frames = [f for f in (download(s) for s in completed) if f is not None]
        cached = pd.concat(frames, ignore_index=True)
        cached.to_parquet(CACHE_PATH)
    current = download(CURRENT_SEASON) if CURRENT_SEASON in ALL_SEASONS else None
    return _trim(pd.concat([cached, current], ignore_index=True) if current is not None else cached)


def _trim(pbp: pd.DataFrame) -> pd.DataFrame:
    """Fifteen-plus seasons is a lot of memory, so each season is slimmed as it arrives. Only 4th downs
    use the play descriptions (to spot fake punts and show real plays), and 32-bit numbers are
    plenty (XGBoost uses them anyway)."""
    pbp.loc[pbp["down"] != 4, "desc"] = None
    return pbp.astype(dict.fromkeys(pbp.select_dtypes("float64").columns, "float32"))


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
        home=home.astype(int),
        won=(margin > 0).astype(int),
        tie=margin == 0,
    )

    # Does the offense get the 2nd-half kickoff? It does if it kicked off to start the game (nflverse's definition).
    opening_kickers = df.groupby("game_id")["defteam"].first()
    df["receive_2h_ko"] = ((df["qtr"] <= 2) & (df["posteam"] == df["game_id"].map(opening_kickers))).astype(int)

    # For the punt model: who has the ball on the next scrimmage play, and where.
    nxt = df.groupby("game_id")[["posteam", "down", "yardline_100", "qtr"]].shift(-1)
    df["next_posteam"], df["next_down"] = nxt["posteam"], nxt["down"]
    df["next_yardline_100"], df["next_qtr"] = nxt["yardline_100"], nxt["qtr"]
    return df


def wp_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Every scrimmage play, labeled with whether the team that had the ball went on to win.
    Ties are dropped (rare, and they don't fit a win/loss label)."""
    rows = df[~df["tie"]].dropna(subset=["posteam_timeouts_remaining", "result"])
    return rows[STATE_COLUMNS + ["won", "season", "home", "receive_2h_ko"]]  # only the columns needed


def conversion_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Real 4th-down attempts (runs and passes, including fakes). nflverse's fourth_down_converted
    already counts touchdowns as conversions, which we checked against the 2025 data."""
    att = df[(df["down"] == 4) & df["play_type"].isin(["run", "pass"])]
    # Fake punts and fake field goals aren't a decision to go for it; they're surprise plays.
    fake = att["desc"].str.contains("Punt formation|Field Goal formation", case=False, na=False)
    return att[~fake].dropna(subset=["posteam_spread"])


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

def fit_models(df: pd.DataFrame, model_seasons: list[int], kick_seasons: list[int], wp_seasons: list[int]) -> dict:
    core = df[df["season"].isin(model_seasons)]

    # 1. Win probability (see WP_FEATURES and WP_PARAMS).
    wp_data = wp_rows(df[df["season"].isin(wp_seasons)])
    wp = xgb.train(WP_PARAMS, xgb.DMatrix(wp_matrix(wp_data).to_numpy(), label=wp_data["won"].to_numpy()), num_boost_round=WP_TREES)

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
    latest = df[df["season"] == max(model_seasons + kick_seasons + wp_seasons)]
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
    m = fit_models(df, [s for s in MODEL_SEASONS if s < TEST_SEASON], [s for s in KICK_SEASONS if s < TEST_SEASON],
                   [s for s in WP_SEASONS if s < TEST_SEASON])
    train = df[df["season"] < TEST_SEASON]
    test = df[df["season"] == TEST_SEASON]
    print(f"\nHoldout evaluation on the {TEST_SEASON} season (models trained only on earlier seasons)\n")
    metrics = {}

    w = wp_rows(test)
    metrics["wp"] = calibration("Win probability", w["won"], wp_predict(m["wp"], w))
    train_rows = wp_rows(train[train["season"].isin(WP_SEASONS)])
    print(f"    overfitting check: log loss {log_loss(train_rows['won'], wp_predict(m['wp'], train_rows)):.4f} on its training seasons")
    try:
        print(f"    benchmark: nflverse's published model scores {log_loss(w['won'], wp_predict(load_nflverse_wp(), w)):.4f} on the same season")
    except Exception as exc:  # the benchmark is optional; training doesn't depend on it
        print(f"    (benchmark skipped: {exc})")

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
    print(f"\nRefitting: win probability on {WP_SEASONS[0]}-{WP_SEASONS[-1]}, conversion on {MODEL_SEASONS[0]}-{MODEL_SEASONS[-1]}, "
          f"field goals and punts on {KICK_SEASONS[0]}-{KICK_SEASONS[-1]}...")
    bundle = fit_models(df, MODEL_SEASONS, KICK_SEASONS, WP_SEASONS)
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
    inches = s["ydstogo"].to_numpy() == 0
    s = s.assign(ydstogo=np.maximum(s["ydstogo"], 1))   # the models only know whole yards
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
        total = 0.0
        for offset, weight in SMOOTHING:  # average over neighboring yard lines (see SMOOTHING)
            spot = np.clip(yl + offset, 1, 99)
            state = pd.DataFrame({
                "yardline_100": spot, "down": down, "score_differential": score,
                "ydstogo": np.minimum(10, spot) if down == 1 else np.minimum(dist, spot),
                "game_seconds_remaining": gsr_after if after_play else gsr,
                "half_seconds_remaining": half_after if after_play else half,
                "posteam_timeouts_remaining": ours if offense_is_us else theirs,
                "defteam_timeouts_remaining": theirs if offense_is_us else ours,
                "posteam_spread": spread if offense_is_us else -spread,
            })
            # Home field and the 2nd-half kickoff aren't calculator inputs, so average over both
            # (receive_2h_ko only matters in the first half; wp_matrix zeroes it after that).
            for home in (0, 1):
                for receive in (0, 1):
                    total = total + weight * 0.25 * wp_predict(b["wp"], state.assign(home=home, receive_2h_ko=receive))
        return total / sum(w for _, w in SMOOTHING)

    def our_first_down(yl, score):
        return wp(yl, score, 1, np.minimum(10, yl), True)

    def their_first_down(yl, our_score):
        yl = np.clip(yl, 1, 99)
        return 1 - wp(yl, -our_score, 1, np.minimum(10, yl), False)  # our WP = 1 - theirs

    # Go for it. Success: first down at the line to gain (a touchdown, then a kickoff, if it's goal
    # to go). Placing the ball exactly at the sticks undersells going for it a little, because real
    # conversions often gain extra yards. Failure: the opponent takes over at the line of scrimmage.
    p_conv = b["conv"].predict_proba(conv_matrix(s))[:, 1]
    p_conv = np.where(inches, 1 - (1 - p_conv) * INCHES_FAIL_SHARE, p_conv)   # see INCHES_FAIL_SHARE
    goal = togo >= y
    wp_conv = np.where(goal, their_first_down(np.full_like(y, b["kickoff_start"]), diff + TD_POINTS),
                       our_first_down(np.maximum(y - togo, 1), diff))
    wp_fail = their_first_down(100 - y, diff)
    go = p_conv * wp_conv + (1 - p_conv) * wp_fail
    # The conservative lean (see FAIL_PENALTY), charged to the failure branch only, so it scales with
    # how likely failing is. "go" stays the honest win probability; the call uses go_call.
    depth = np.clip((y - OWN_HALF_LINE) / (99 - OWN_HALF_LINE), 0, 1)  # 0 at your own 45, 1 at your own 1
    early = np.clip((gsr - 900) / 900, 0, 1)                           # 1 in the first half, 0 by the 4th quarter
    penalty = FAIL_PENALTY * depth * early
    go_call = p_conv * wp_conv + (1 - p_conv) * (wp_fail - penalty)

    # Field goal. Make: up 3 more, then kick off. Miss: NFL rules give the opponent the ball at the
    # spot of the kick (8 yards behind the line), or at their 20 if that spot is inside the 20.
    p_make = predict_fg(b, s)
    wp_make = their_first_down(np.full_like(y, b["kickoff_start"]), diff + 3)
    kick_spot = y + FG_OFFSET - 10
    wp_miss = their_first_down(np.minimum(100 - kick_spot, 80), diff)
    fg = p_make * wp_make + (1 - p_make) * wp_miss
    # The take-the-points lean (see FG_BONUS), scaled by the chance the kick is good.
    in_range = np.clip((FG_RANGE_LINE - y) / FG_RAMP, 0, 1)   # 0 at the opponent's 44, full from the 41 in
    in_range = in_range * ~(goal & (y <= 3))                 # but never on 4th & goal from the 3 or closer
    fg_call = fg + FG_BONUS * p_make * in_range * early

    # Punt: the opponent starts wherever punts from this spot usually leave them.
    landing = sum(w * b["punt"].predict(pd.DataFrame({"yardline_100": np.clip(y + o, 1, 99)})) for o, w in SMOOTHING)
    opp_start = np.clip(landing / sum(w for _, w in SMOOTHING), 1, 99)
    punt = their_first_down(opp_start, diff)

    out = pd.DataFrame({
        "go": go, "fg": fg, "punt": punt, "go_call": go_call, "fg_call": fg_call, "lean": penalty,
        "wp_now": wp(y, diff, 4, togo, True, after_play=False),   # before the decision, for context
        "p_convert": p_conv, "wp_convert": wp_conv, "wp_fail": wp_fail,
        "p_make": p_make, "kick_distance": y + FG_OFFSET, "wp_make": wp_make, "wp_miss": wp_miss,
        "punt_opp_yardline_100": opp_start,
    })
    vals = np.c_[go_call, fg_call, punt]  # decide with both leans applied
    order = np.argsort(-vals, axis=1)
    rows = np.arange(len(out))
    out["best"] = np.array(OPTIONS)[order[:, 0]]
    out["margin"] = vals[rows, order[:, 0]] - vals[rows, order[:, 1]]   # how much better than the runner-up
    out["runner_up"] = np.array(OPTIONS)[order[:, 1]]
    lean_active = (penalty > 0) | (FG_BONUS * p_make * in_range * early > 0)
    out["toss_up"] = (out["margin"] < TOSS_UP) & ~lean_active     # see TOSS_UP

    # Break-even: the conversion rate at which going for it exactly ties the better of kicking and
    # punting. Solve p * wp_conv + (1 - p) * wp_fail = best_alternative for p. This turns "the
    # model says go" into something checkable: "you need 31%, and teams convert 55% of these".
    with np.errstate(divide="ignore", invalid="ignore"):
        breakeven = (np.maximum(fg_call, punt) - (wp_fail - penalty)) / (wp_conv - (wp_fail - penalty))
    out["breakeven"] = np.where(wp_conv > wp_fail - penalty, np.clip(breakeven, 0, 1), np.nan)
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
    ("4th & 3 at your own 11, tied, 1st quarter",
     dict(yardline_100=89, ydstogo=3, score_differential=0, qtr=1, quarter_seconds_remaining=600), "punt"),
    ("4th & 5 at your own 25, tied, 1st quarter",
     dict(yardline_100=75, ydstogo=5, score_differential=0, qtr=1, quarter_seconds_remaining=600), "punt"),
    ("4th & 1 at your own 45, tied, 1st quarter",
     dict(yardline_100=55, ydstogo=1, score_differential=0, qtr=1, quarter_seconds_remaining=600), "go"),
    ("4th & 2 at the opp 45, tied, 1st quarter",
     dict(yardline_100=45, ydstogo=2, score_differential=0, qtr=1, quarter_seconds_remaining=600), "go"),
    ("4th & 2 at your own 16, tied, 1st quarter: too risky this early",
     dict(yardline_100=84, ydstogo=2, score_differential=0, qtr=1, quarter_seconds_remaining=600), "punt"),
    ("4th & 5 at your own 30, down 7, 1:30 left: no time to be careful",
     dict(yardline_100=70, ydstogo=5, score_differential=-7, qtr=4, quarter_seconds_remaining=90), "go"),
    ("4th & 4 at the opp 25, tied, 1st quarter: take the points",
     dict(yardline_100=25, ydstogo=4, score_differential=0, qtr=1, quarter_seconds_remaining=600), "fg"),
    ("4th & 1 at the opp 20, tied, 1st quarter",
     dict(yardline_100=20, ydstogo=1, score_differential=0, qtr=1, quarter_seconds_remaining=600), "go"),
    ("4th & 8 at the opp 41, tied, 5:00 left in the 2nd: a 59-yard try beats punting",
     dict(yardline_100=41, ydstogo=8, score_differential=0, qtr=2, quarter_seconds_remaining=300), "fg"),
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
    spots = [{**base, "yardline_100": yl} for yl in range(2, 100, 7)]
    inch = evaluate(b, states_frame([{**row, "ydstogo": 0} for row in spots]))
    yard = evaluate(b, states_frame([{**row, "ydstogo": 1} for row in spots]))
    inch_ok = bool(np.all(inch["go"].to_numpy() >= yard["go"].to_numpy() - 1e-9))
    print(f"{'PASS' if inch_ok else 'FAIL'}  going for it on 4th & inches is never worse than on 4th & 1")
    late = evaluate(b, states_frame([{**base, "yardline_100": yl, "ydstogo": d} for yl in range(5, 96, 5) for d in (1, 3, 5)]))
    early = evaluate(b, states_frame([{**base, "qtr": 1, "quarter_seconds_remaining": 600, "yardline_100": yl, "ydstogo": d}
                                      for yl in list(range(5, 41, 5)) + list(range(60, 96, 5)) for d in (1, 3, 5)]))
    toss_ok = bool((late["toss_up"] == (late["margin"] < TOSS_UP)).all() and not early["toss_up"].any())
    print(f"{'PASS' if toss_ok else 'FAIL'}  toss-ups only where no lean is in play, and always when the call is that close")
    return ok and conv_ok and fg_ok and inch_ok and toss_ok


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
    booster = model if isinstance(model, xgb.Booster) else model.get_booster()
    raw = json.loads(booster.save_raw(raw_format="json"))
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
    margins = booster.predict(xgb.DMatrix(probe.to_numpy(np.float32), feature_names=booster.feature_names), output_margin=True)
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
        "wp": _browser_trees(b["wp"], wp_matrix(probe.assign(home=rng.integers(0, 2, n), receive_2h_ko=rng.integers(0, 2, n)))),
        "conv": _browser_trees(b["conv"], conv_matrix(probe)),
        "punt": _browser_trees(b["punt"], probe[["yardline_100"]]),
        "fg": {"intercept": float(b["fg"].intercept_[0]), "coef": [float(c) for c in b["fg"].coef_[0]], "max": b["fg_max"]},
        "kickoff_start": b["kickoff_start"],
        "constants": {"play_seconds": PLAY_SECONDS, "td_points": TD_POINTS, "fg_offset": FG_OFFSET, "smoothing": SMOOTHING, "red_zone": RED_ZONE,
                      "fail_penalty": FAIL_PENALTY, "own_half_line": OWN_HALF_LINE,
                      "fg_bonus": FG_BONUS, "fg_range_line": FG_RANGE_LINE, "fg_ramp": FG_RAMP,
                      "inches_fail_share": INCHES_FAIL_SHARE,
                      "toss_up": TOSS_UP},
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
    ydstogo: int = Field(ge=0, le=99, description="Yards to go; 0 means inches")

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
        "best": r["best"], "toss_up": bool(r["toss_up"]), "margin": f("margin"), "wp_now": f("wp_now"), "breakeven": f("breakeven"),
        "options": {
            "go": {"wp": f("go"), "wp_with_lean": f("go_call"), "p_convert": f("p_convert"), "wp_convert": f("wp_convert"), "wp_fail": f("wp_fail")},
            "fg": {"wp": f("fg"), "wp_with_lean": f("fg_call"), "p_make": f("p_make"), "kick_distance": f("kick_distance"),
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