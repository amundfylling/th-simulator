"""M10 live mode: does updating strengths from the day's games improve in-event forecasts?

The website's live mode fixes the games already played and simulates the rest. Until now
the rest was played with the pre-event strengths; with same-day form the engine conditions
every strength on the games already played (web/src/engine/dayForm.ts). This backtest replays
the group-then-playoff events of each window from two points in the day:

  half   the first half of the group games (in order) are played
  ko     all group games are played: the forecast at the playoff start

and scores the final placings (winner log loss, RPS, top-3 Brier) with the update switched
off (`_day0`) and on with a few values of the per-game noise sd (the Kalman rating's is 2.6
goals). Players, LightGBM pair shifts, game model and form shock are the shipped M9 setup.

Run (after python -m thsim.research.m9_tournaments for the window):
    python -m thsim.research.m10_live --window select
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from thsim.backtest_tournaments import WORK, score
from thsim.data.loading import load_games_cached
from thsim.research.m9_tournaments import PRE_JSON
from thsim.research.puck_tournaments import BASE
from thsim.structure import structural

OBS_SDS = ["0", "2", "2.6", "3.4"]


def group_games(games: pd.DataFrame, uids: set[int]) -> dict[int, list[dict]]:
    g = structural(games)
    g = g.loc[g["tournament_uid"].isin(uids) & g["stage_kind"].eq("rr") & ~g["walkover"]]
    out: dict[int, list[dict]] = {}
    for r in g.itertuples():
        out.setdefault(int(r.tournament_uid), []).append(
            {"stage": "groups", "p1": int(r.p1), "p2": int(r.p2), "g1": int(r.g1), "g2": int(r.g2), "ot": bool(r.ot)})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", default="select", choices=["select", "final"])
    parser.add_argument("--runs", type=int, default=2000)
    parser.add_argument("--obs", nargs="*", default=OBS_SDS)
    args = parser.parse_args()
    win = args.window
    events = json.loads((WORK / f"events_{win}_gbm.json").read_text())
    events["events"] = [e for e in events["events"] if e["kind"] == "groups_ko"]
    played = group_games(load_games_cached(), {e["tournament_uid"] for e in events["events"]})
    variants = [f"{BASE}_day{o}" for o in args.obs]
    res: dict = {"window": win, "events": len(events["events"])}
    for point in ("half", "ko"):
        tag = f"{win}_live_{point}"
        evs = json.loads(json.dumps(events))
        for e in evs["events"]:
            gl = played.get(e["tournament_uid"], [])
            e["played"] = gl[: len(gl) // 2] if point == "half" else gl
        ev_path = WORK / f"events_{tag}.json"
        ev_path.write_text(json.dumps(evs))
        subprocess.run(["npx", "tsx", "scripts/backtest.ts", str(ev_path.resolve()),
                        str((WORK / f"model_{win}_composite+h2h.json").resolve()), str((WORK / f"pred_{tag}.json").resolve()),
                        str(args.runs), *variants],
                       cwd="web", check=True, env=os.environ | {"GBM_MODEL": str(PRE_JSON.resolve())})
        out = {}
        for subset, only_ok in (("all", False), ("rule_ok", True)):
            sc = score(tag, only_rule_ok=only_ok)
            summ = sc["summary"]
            df = sc["rows"]
            off = f"{BASE}_day0"
            paired = {}
            for v in variants[1:]:
                paired[v] = {}
                for m in ("winner_ll", "rps", "top3_brier"):
                    pv = df.pivot(index="uid", columns="variant", values=m)
                    d = pv[v] - pv[off]
                    paired[v][m] = {"mean_diff": float(d.mean()), "t_p": float(stats.ttest_1samp(d, 0).pvalue),
                                    "better_share": float((d < 0).mean())}
            geo = df.groupby("variant")["winner_ll"].mean().map(lambda x: float(np.exp(-x)))
            out[subset] = {"summary": summ.round(5).to_dict(orient="index"), "paired": paired, "geo_p_winner": geo.to_dict()}
            print(win, point, subset)
            print(summ[["events", "winner_ll", "rps", "top3_brier", "fav_won"]].round(4).to_string())
            for v, pm in paired.items():
                print("  ", v, {m: (round(x["mean_diff"], 5), round(x["t_p"], 4)) for m, x in pm.items()})
        res[point] = out
    Path(f"reports/m10_live_{win}.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
