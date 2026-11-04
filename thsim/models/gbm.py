"""LightGBM layer (M9): pre-event win/draw/loss odds for any pair of players.

Trained on frozen pre-event features (``features.build(freeze=True)``) with every game in
both orientations, so the model cannot see player order. The website and the tournament
backtest turn its odds into a per-pair shift on top of the composite rating gap (see
``web/src/engine/gbm.ts``), through the engine's head-to-head map.

Current per-player features come from "probe" rows: one walkover per player on the export
date against a brand-new opponent. Walkovers update nothing, so each probe row's pre-game
features are the player's state on that date, computed by exactly the code that built
the training features. The brand-new opponent's side gives the newcomer feature vector.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from thsim.backtest_games import Context
from thsim.ratings import composite as C
from thsim.ratings import models as M
from thsim.research import features as FT

FEATURES = ["d", "sd", "h2h", "meet", "po", "level", "bordshockey", "team"] + [
    f"{c}_{s}" for c in FT.SIDE_COLS for s in (1, 2)]
PARAMS = {"objective": "multiclass", "num_class": 3, "learning_rate": 0.05, "num_leaves": 63,
          "min_data_in_leaf": 400, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
          "lambda_l2": 5.0, "verbose": -1, "num_threads": 4, "seed": 7}
SETTINGS_PATH = Path("reports/m9_params.json")
FIT_YEARS = 10


def both_orientations(f: pd.DataFrame, y: np.ndarray, mask: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    a = f.loc[mask, FEATURES]
    b = FT.mirror(f.loc[mask])[FEATURES]
    return pd.concat([a, b], ignore_index=True), np.r_[y[mask], 2 - y[mask]]


def train(f: pd.DataFrame, y: np.ndarray, train_mask: np.ndarray, valid_mask: np.ndarray | None = None,
          rounds: int | None = None, log=print):
    """Early-stopped on ``valid_mask``, or a fixed number of ``rounds``."""
    import lightgbm as lgb

    Xt, yt = both_orientations(f, y, train_mask)
    t0 = time.time()
    if valid_mask is not None:
        Xv, yv = both_orientations(f, y, valid_mask)
        model = lgb.train(PARAMS, lgb.Dataset(Xt, yt), num_boost_round=3000, valid_sets=[lgb.Dataset(Xv, yv)],
                          callbacks=[lgb.early_stopping(50, verbose=False)])
    else:
        model = lgb.train(PARAMS, lgb.Dataset(Xt, yt), num_boost_round=rounds)
        model.best_iteration = rounds
    log(f"gbm: {model.best_iteration} rounds on {len(yt):,} rows, {time.time() - t0:.0f}s")
    return model


def predict(model, f: pd.DataFrame) -> np.ndarray:
    """Order-free odds (A wins, draw, B wins): the prediction averaged with the mirrored one."""
    it = model.best_iteration or None
    p = model.predict(f[FEATURES], num_iteration=it)
    q = model.predict(FT.mirror(f)[FEATURES], num_iteration=it)
    return (p + q[:, ::-1]) / 2


def to_json(model) -> dict:
    """Compact tree arrays for the TypeScript evaluator (web/src/engine/gbm.ts).

    Per tree: split feature, threshold, children (>= 0: node index, < 0: ~leaf index),
    missing handling (0 none, 1 zero, 2 NaN), default-left, and leaf values. Trees are
    class-interleaved: tree k adds to class k % 3.
    """
    dump = model.dump_model(num_iteration=model.best_iteration or None)
    trees = []
    for t in dump["tree_info"]:
        feat, thr, left, right, miss, dl, leaves = [], [], [], [], [], [], []

        def walk(node) -> int:
            if "leaf_value" in node:
                leaves.append(round(float(node["leaf_value"]), 6))
                return ~(len(leaves) - 1)
            k = len(feat)
            feat.append(int(node["split_feature"]))
            thr.append(float(node["threshold"]))
            miss.append({"None": 0, "Zero": 1, "NaN": 2}[node["missing_type"]])
            dl.append(1 if node["default_left"] else 0)
            left.append(0)
            right.append(0)
            left[k] = walk(node["left_child"])
            right[k] = walk(node["right_child"])
            return k

        root = walk(t["tree_structure"])
        trees.append({"f": feat, "t": thr, "l": left, "r": right, "m": miss, "dl": dl, "v": leaves, "root": root})
    return {"features": FEATURES, "side": FT.SIDE_COLS, "num_class": 3, "trees": trees}


# --- current features ------------------------------------------------------------------


def probe_games(games: pd.DataFrame, as_of: pd.Timestamp, player_ids) -> pd.DataFrame:
    """One walkover per player on ``as_of`` against a single brand-new opponent."""
    ids = np.asarray(sorted(player_ids))
    newcomer = int(min(games["p1"].min(), games["p2"].min(), 0)) - 1
    tmpl = games.iloc[[0]].copy()
    probe = pd.concat([tmpl] * len(ids), ignore_index=True)
    probe["p1"] = ids
    probe["p2"] = newcomer
    probe["g1"] = 0
    probe["g2"] = 0
    probe["walkover"] = True
    probe["date"] = as_of
    probe["stage_kind"] = "rr"
    probe["ot"] = False
    probe["team_event"] = False
    probe["level"] = np.nan
    probe["source"] = "scorpion"
    uid0 = int(games["tournament_uid"].max()) + 1
    probe["tournament_uid"] = uid0 + np.arange(len(ids))  # each probe is its own event
    for col in ("p1_name", "p2_name"):
        if col in probe:
            probe[col] = None
    return probe


def current_features(games: pd.DataFrame, players: pd.DataFrame, kp: M.KalmanParams, cp: C.CompositeParams,
                     as_of: pd.Timestamp, player_ids) -> tuple[pd.DataFrame, dict]:
    """Per-player frozen side features on ``as_of``, and the newcomer vector."""
    probe = probe_games(games, as_of, player_ids)
    allg = pd.concat([games, probe], ignore_index=True)
    ctx = Context(allg, players)
    f = FT.build(ctx, allg, kp, cp, freeze=True)
    rows = f.iloc[len(games):]
    side = pd.DataFrame({c: rows[f"{c}_1"].to_numpy() for c in FT.SIDE_COLS}, index=probe["p1"].to_numpy())
    newcomer = {c: float(rows[f"{c}_2"].iloc[0]) for c in FT.SIDE_COLS}
    return side, newcomer


def save_settings(rounds: int, extra: dict) -> None:
    SETTINGS_PATH.write_text(json.dumps({"rounds": rounds, "params": PARAMS, "features": FEATURES, **extra}, indent=1))


def load_settings() -> dict | None:
    return json.loads(SETTINGS_PATH.read_text()) if SETTINGS_PATH.exists() else None
