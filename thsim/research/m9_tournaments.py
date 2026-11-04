"""M9 tournament level: the LightGBM layer's pre-event odds fed into the simulator.

A pre-tournament forecast only knows what stood before the event's first game, so the model
is trained on *frozen* features (head-to-head, form, games played as they stood at each
player's first game of the event; no in-event form), on 2015-2020 with 2021 for early
stopping, and the same model serves both windows. Every entrant carries their frozen feature
vector; the engine-side code the website runs (web/src/engine/gbm.ts) turns the model's odds
for every pair into a per-pair shift on top of the composite rating gap:

    shift = inv(p) / k - (mu_a - mu_b)        inv(p) = (logit pA - logit pB) / (2 beta)

where k = 1 / sqrt(1 + pi beta^2 sd^2 / 8) undoes the flattening the engine's rating draws
add. Events, engine, game model and form shock 0.45 are the M8 setup.

Run (after python -m thsim.research.puck_tournaments for the window):
    python -m thsim.research.m9_tournaments --window select [--retrain]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from thsim.backtest_games import TUNE, Context
from thsim.backtest_tournaments import WORK, score
from thsim.data.loading import load_games_cached, load_players
from thsim.models import gbm as G
from thsim.ratings import composite as C
from thsim.research import features as FT
from thsim.research.puck import kalman_params
from thsim.research.puck_tournaments import BASE, FORM, pair_history

PRE_MODEL = WORK / "m9_gbm_pre.txt"
PRE_JSON = WORK / "m9_gbm_pre.json"


def player_snapshots(games: pd.DataFrame, f_fz: pd.DataFrame) -> pd.DataFrame:
    """(tournament_uid, player) -> frozen side features at the player's first game of the event."""
    parts = []
    for s, pcol in ((1, "p1"), (2, "p2")):
        cols = {f"{c}_{s}": c for c in FT.SIDE_COLS}
        part = f_fz[list(cols)].rename(columns=cols)
        part["t"] = games["tournament_uid"].to_numpy()
        part["p"] = games[pcol].to_numpy()
        part["i"] = np.arange(len(games))
        parts.append(part)
    return pd.concat(parts).sort_values("i").drop_duplicates(["t", "p"]).set_index(["t", "p"])


def event_h2h(hist: dict, ids: list[int], start: int, cp: C.CompositeParams) -> dict[str, list[float]]:
    """[edge, meetings] for pairs the website gets (2+ meetings, |edge| >= 0.01), before the event."""
    out = {}
    ids = sorted(ids)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            h = hist.get((a, b))
            if h is None:
                continue
            n = int(np.searchsorted(h[0], start))
            if n < C.H2H_MIN_MEETINGS:
                continue
            e = cp.h2h_coef * h[1][n - 1] / (n + cp.h2h_prior)
            if abs(e) >= C.H2H_MIN_ABS:
                out[f"{a}-{b}"] = [round(float(e), 4), n]
    return out


def _clean(v: float) -> float | None:
    return None if not np.isfinite(v) else round(float(v), 5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", default="select", choices=["select", "final"])
    parser.add_argument("--runs", type=int, default=2000)
    parser.add_argument("--retrain", action="store_true")
    args = parser.parse_args()
    win = args.window
    games = load_games_cached()
    ctx = Context(games, load_players())
    kp, cp = kalman_params(), C.load()
    f_fz = FT.build(ctx, games, kp, cp, freeze=True)
    if args.retrain or not PRE_MODEL.exists():
        use = ctx.known & (ctx.rr | ctx.po) & ctx.window(TUNE)
        model = G.train(f_fz, ctx.y3, use & (ctx.date < np.datetime64("2021-01-01")), use & (ctx.date >= np.datetime64("2021-01-01")))
        model.save_model(str(PRE_MODEL), num_iteration=model.best_iteration)
        PRE_JSON.write_text(json.dumps(G.to_json(model), separators=(",", ":")))
        G.save_settings(model.best_iteration, {"chosen_on": "2015-2020 training, 2021 early stopping (frozen features)"})
    snap = player_snapshots(games, f_fz)
    r = C.run(ctx.prep, kp, cp)
    hist = pair_history(ctx.prep, C.residuals(ctx.prep, r, cp), games)
    first = games.reset_index().groupby("tournament_uid")["index"].min()
    src = games.groupby("tournament_uid")["source"].first()
    lvl = games.groupby("tournament_uid")["level"].first()
    events = json.loads((WORK / f"events_{win}_composite+h2h.json").read_text())
    for ev in events["events"]:
        uid = ev["tournament_uid"]
        side = snap.loc[uid]
        for p in ev["players"]:
            p["feat"] = [_clean(v) for v in side.loc[p["id"], FT.SIDE_COLS].to_numpy(float)]
        ev["h2h"] = event_h2h(hist, [p["id"] for p in ev["players"]], int(first[uid]), cp)
        level = lvl.get(uid)
        ev["ctx"] = {"level": float(level) if pd.notna(level) else -1.0, "bordshockey": float(src.get(uid) == "bordshockey")}
    ev_path = WORK / f"events_{win}_gbm.json"
    ev_path.write_text(json.dumps(events))
    part = WORK / f"pred_{win}_gbm.json"
    subprocess.run(["npx", "tsx", "scripts/backtest.ts", str(ev_path.resolve()),
                    str((WORK / f"model_{win}_composite+h2h.json").resolve()), str(part.resolve()), str(args.runs), BASE],
                   cwd="web", check=True, env=os.environ | {"GBM_MODEL": str(PRE_JSON.resolve())})
    merged = json.loads((WORK / f"pred_{win}p.json").read_text())
    merged["variants"][f"gbm_form{FORM}"] = json.loads(part.read_text())["variants"][BASE]
    tag = f"{win}g"
    shutil.copy(WORK / f"events_{win}.json", WORK / f"events_{tag}.json")
    (WORK / f"pred_{tag}.json").write_text(json.dumps(merged))
    res = score(tag)
    print(res["summary"].round(4).to_string())
    df = res["rows"]
    paired = {}
    for m in ("winner_ll", "rps", "top3_brier"):
        pv = df.pivot(index="uid", columns="variant", values=m)
        for other in (BASE, f"composite+h2h_form{FORM}"):
            dlt = pv[f"gbm_form{FORM}"] - pv[other]
            paired[f"{m}: gbm vs {other}"] = {"mean_diff": float(dlt.mean()), "t_p": float(stats.ttest_1samp(dlt, 0).pvalue),
                                              "better_share": float((dlt < 0).mean())}
            print(m, other, round(dlt.mean(), 5), round(stats.ttest_1samp(dlt, 0).pvalue, 4), round((dlt < 0).mean(), 3))
    Path(f"reports/m9_tournaments_{win}.json").write_text(json.dumps(
        {"window": win, "variants": res["summary"].round(4).to_dict(orient="index"), "paired": paired},
        indent=1, default=float))


if __name__ == "__main__":
    main()
