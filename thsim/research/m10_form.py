"""M10: does same-day form predict the next games of the event?

Tournaments are played in a day or two, so how a player performs earlier in the event might
say something about their later games (playoffs above all) beyond what the game-by-game rating
updates already capture. Everything is measured against expectations from each player's
rating *before the event*, so "form" means doing better or worse than the day was expected
to go, not just the rating moving.

Variants of the M9 LightGBM layer (live features, trained 2015-2020, early-stopped on 2021,
scored on 2022-2024 and 2025-2026):

  A  no same-event features
  B  M9 as shipped: earlier games in the event and the goal residual vs the live gap
  C  B + richer same-day signals: score and goals vs pre-event expectation (shrunk), the
     last three games' residual, how far the rating already moved during the event, the
     opponent-adjusted points share, and the stage (playoff after a group stage)

Run: python -m thsim.research.m10_form [--final]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from thsim.backtest_games import FINAL, SELECT, TUNE, Context
from thsim.data.loading import load_games_cached, load_players
from thsim.models import gbm as G
from thsim.ratings import composite as C
from thsim.research import features as FT
from thsim.research import hypotheses as H
from thsim.research.puck import kalman_params

EV = ["ev_games", "ev_form"]
RICH = ["day_score", "day_goals", "day_last3", "day_drift"]


def same_day(games: pd.DataFrame, f: pd.DataFrame, cp: C.CompositeParams, prep) -> pd.DataFrame:
    """Per game and side: performance earlier in the same event vs pre-event expectations."""
    n = len(games)
    t = games["tournament_uid"].to_numpy()
    long = pd.DataFrame({
        "t": np.r_[t, t], "p": np.r_[games["p1"].to_numpy(), games["p2"].to_numpy()],
        "i": np.r_[np.arange(n), np.arange(n)], "side": np.r_[np.zeros(n, int), np.ones(n, int)],
        "mu": np.r_[f["mu_1"].to_numpy(), f["mu_2"].to_numpy()],
    })
    long = long.sort_values(["i", "side"], kind="stable")
    long["mu0"] = long.groupby(["t", "p"])["mu"].transform("first")  # rating before the event
    mu0 = long.sort_values(["side", "i"])["mu0"].to_numpy()
    mu0_1, mu0_2 = mu0[:n], mu0[n:]
    d0 = mu0_1 - mu0_2
    var = f["sd"].to_numpy() ** 2
    exp0 = C.expected_score(d0, var, cp.exp_beta, cp.exp_c)
    played = ~prep.skip
    s_res = np.where(played, prep.score - exp0, 0.0)
    g_res = np.where(played, np.clip(prep.gd, -6, 6) - d0, 0.0)
    long = pd.DataFrame({
        "t": np.r_[t, t], "p": np.r_[games["p1"].to_numpy(), games["p2"].to_numpy()],
        "i": np.r_[np.arange(n), np.arange(n)], "side": np.r_[np.zeros(n, int), np.ones(n, int)],
        "g": np.r_[played, played].astype(float), "s": np.r_[s_res, -s_res], "gr": np.r_[g_res, -g_res],
        "mu": np.r_[f["mu_1"].to_numpy(), f["mu_2"].to_numpy()], "mu0": np.r_[mu0_1, mu0_2],
    }).sort_values(["i", "side"], kind="stable")
    grp = long.groupby(["t", "p"], sort=False)
    nb = grp["g"].cumsum() - long["g"]
    long["day_score"] = (grp["s"].cumsum() - long["s"]) / (nb + 3)
    long["day_goals"] = (grp["gr"].cumsum() - long["gr"]) / (nb + 3)
    last3 = grp["gr"].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean()).fillna(0.0)
    long["day_last3"] = last3
    long["day_drift"] = long["mu"] - long["mu0"]
    long = long.sort_values(["side", "i"])
    out = pd.DataFrame(index=f.index)
    for c in RICH:
        v = long[c].to_numpy()
        out[f"{c}_1"], out[f"{c}_2"] = v[:n], v[n:]
    return out


def run(final: bool) -> dict:
    games = load_games_cached()
    ctx = Context(games, load_players())
    kp, cp = kalman_params(), C.load()
    f = FT.build(ctx, games, kp, cp)
    rich = same_day(games, f, cp, ctx.prep)
    f = pd.concat([f, rich], axis=1)
    use = ctx.known & (ctx.rr | ctx.po)
    tune = use & ctx.window(TUNE)
    train = tune & (ctx.date < np.datetime64("2021-01-01"))
    valid = tune & (ctx.date >= np.datetime64("2021-01-01"))
    windows = {"select": use & ctx.window(SELECT)}
    if final:
        windows["final"] = use & ctx.window(FINAL)
    side_ev = [f"{c}_{s}" for c in EV for s in (1, 2)]
    side_rich = [f"{c}_{s}" for c in RICH for s in (1, 2)]
    variants = {
        "A no same-event features": [c for c in G.FEATURES if c not in side_ev],
        "B M9 as shipped": list(G.FEATURES),
        "C + richer same-day form": list(G.FEATURES) + side_rich,
    }
    # The mirror helper only knows the M9 columns: extend it for the new side columns.
    FT.SIDE_COLS.extend(c for c in RICH if c not in FT.SIDE_COLS)
    y = ctx.y3
    po = ctx.po
    ev_n = np.minimum(f["ev_games_1"].to_numpy(), f["ev_games_2"].to_numpy())
    tour = games["tournament_uid"].to_numpy()
    preds = {}
    old = list(G.FEATURES)
    res: dict = {"importance": {}}
    try:
        for name, cols in variants.items():
            G.FEATURES[:] = cols
            model = G.train(f, y, train, valid)
            preds[name] = G.predict(model, f)
            imp = pd.Series(model.feature_importance("gain"), index=cols)
            res["importance"][name] = {k: round(float(v), 4) for k, v in (imp / imp.sum()).items()
                                       if k.startswith(("ev_", "day_"))}
    finally:
        G.FEATURES[:] = old
    base = "B M9 as shipped"
    for wname, mask in windows.items():
        idx = np.flatnonzero(mask)
        yy = y[idx]
        ll = {k: -np.log(np.clip(p[idx][np.arange(len(idx)), yy], 1e-12, 1)) for k, p in preds.items()}
        parts = {
            "all": np.ones(len(idx), bool),
            "playoff after 3+ games each": po[idx] & (ev_n[idx] >= 3),
            "round-robin, 5+ games each": ~po[idx] & (ev_n[idx] >= 5),
            "first game of the event": ev_n[idx] == 0,
        }
        out = {}
        for name, l in ll.items():
            out[name] = {}
            for part, pm in parts.items():
                diff = l - ll[base]
                out[name][part] = {"n": int(pm.sum()), "ll": float(l[pm].mean()), "delta_x1000": float(diff[pm].mean() * 1000),
                                   "se_x1000": H.cluster_se(diff[pm], tour[idx][pm]) * 1000}
        res[wname] = out
        for name, o in out.items():
            print(wname, name, {k: (round(v["ll"], 4), round(v["delta_x1000"], 2)) for k, v in o.items()}, flush=True)
    return res


def report(res: dict, final: bool) -> str:
    wins = ["select", "final"] if final else ["select"]
    lines = ["# M10: same-day form (game level)", "",
             "Variants of the M9 LightGBM layer, all trained 2015-2020 and early-stopped on 2021. 3-way regulation log",
             "loss; Δ ×1000 vs M9 as shipped (negative = better), SE clustered by tournament. Form is measured against",
             "each player's rating before the event.", ""]
    for w in wins:
        parts = list(next(iter(res[w].values())).keys())
        lines += [f"## {w}", "", "| variant | " + " | ".join(f"{p} (n={res[w]['B M9 as shipped'][p]['n']:,})" for p in parts) + " |",
                  "|---" * (len(parts) + 1) + "|"]
        for name, o in res[w].items():
            lines.append(f"| {name} | " + " | ".join(f"{o[p]['ll']:.4f} ({o[p]['delta_x1000']:+.2f} ±{o[p]['se_x1000']:.2f})" for p in parts) + " |")
        lines.append("")
    lines += ["## Share of LightGBM gain from same-event features", ""]
    for name, imp in res["importance"].items():
        lines.append(f"- {name}: " + (", ".join(f"{k} {v:.3f}" for k, v in imp.items()) or "none"))
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("reports/m10_form.md"))
    args = parser.parse_args()
    res = run(args.final)
    args.out.write_text(report(res, args.final))
    args.out.with_suffix(".json").write_text(json.dumps(res, indent=1))
    print(args.out.read_text())


if __name__ == "__main__":
    main()
