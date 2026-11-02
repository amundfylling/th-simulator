"""Tournament-level before/after for the composite rating (h2h.puck.no ideas).

Same events, engine, form shock (0.45) and scoring as the M5/M7 tournament backtests. For
each window the game model is refitted on the composite ratings with games up to the end
of the previous window (2021 for 2022-2024, 2024 for 2025-2026). Rating settings, blend
weight and head-to-head coefficient were all fitted on 2015-2021. Head-to-head edges are
frozen at each event's start: no edge sees a result from the event being forecast.

Run (after ``python -m thsim.backtest_tournaments build`` and ``simulate --variants
kalman_form0.45`` for the window):
    python -m thsim.research.puck_tournaments --window select
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

from thsim.backtest_games import Context
from thsim.backtest_tournaments import WORK, _pre_event_ratings, score
from thsim.data.loading import load_games_cached, load_players
from thsim.fit_game_model import load_params
from thsim.models import game_model
from thsim.ratings import composite as C

FIT_END = {"select": "2021-12-31", "final": "2024-12-31"}
FORM = 0.45
BASE = f"kalman_form{FORM}"


def pair_history(prep, resid: np.ndarray, games) -> dict:
    """(lo id, hi id) -> (game indices, cumulative residual from lo's view)."""
    ok = np.flatnonzero(~prep.skip)
    pa, pb = prep.players[prep.a[ok]], prep.players[prep.b[ok]]
    lo, hi = np.minimum(pa, pb), np.maximum(pa, pb)
    r = np.where(pa == lo, 1.0, -1.0) * resid[ok]
    buckets: dict = defaultdict(lambda: ([], []))
    for i, x, y, v in zip(ok, lo, hi, r):
        b = buckets[(int(x), int(y))]
        b[0].append(i)
        b[1].append(v)
    return {k: (np.asarray(ix), np.cumsum(v)) for k, (ix, v) in buckets.items()}


def event_edges(hist: dict, ids: list[int], start: int, cp: C.CompositeParams) -> dict[str, float]:
    """Edges among an event's entrants from games before the event's first game."""
    out = {}
    ids = sorted(ids)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            h = hist.get((a, b))
            if h is None:
                continue
            n = int(np.searchsorted(h[0], start))
            # Same filters as the website export (C.h2h_final): 2+ meetings, |edge| >= 0.01.
            if n < C.H2H_MIN_MEETINGS:
                continue
            e = cp.h2h_coef * h[1][n - 1] / (n + cp.h2h_prior)
            if abs(e) >= C.H2H_MIN_ABS:
                out[f"{a}-{b}"] = round(float(e), 4)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", default="select", choices=list(FIT_END))
    parser.add_argument("--runs", type=int, default=2000)
    args = parser.parse_args()
    win = args.window
    games = load_games_cached()
    ctx = Context(games, load_players())
    kp = load_params()
    cp = C.load()
    r = C.run(ctx.prep, kp, cp)
    d = (r["mu1"] - r["mu2"]).to_numpy()
    var = (r["var1"] + r["var2"]).to_numpy()
    resid = C.residuals(ctx.prep, r, cp)
    event = games["tournament_uid"].factorize()[0]
    edge, _ = C.h2h_edge(ctx.prep, resid, event, cp.h2h_prior)
    fit_span = ("2015-01-01", FIT_END[win])
    gm_rows = ctx.window(fit_span) & ctx.known & ~games["walkover"].to_numpy()
    age_years = (np.datetime64(FIT_END[win]) - games["date"].to_numpy()).astype("timedelta64[D]").astype(float) / 365.25
    weights = 0.5 ** (np.maximum(age_years, 0) / 3.0)
    g1, g2, ot = games["g1"].to_numpy(), games["g2"].to_numpy(), games["ot"].to_numpy()
    po, src = games["stage_kind"].eq("po").to_numpy(), games["source"].to_numpy()
    pre = _pre_event_ratings(games, {"mu": (r["mu1"].to_numpy(), r["mu2"].to_numpy()),
                                     "var": (r["var1"].to_numpy(), r["var2"].to_numpy())})
    first = games.reset_index().groupby("tournament_uid")["index"].min()
    hist = pair_history(ctx.prep, resid, games)

    events = json.loads((WORK / f"events_{win}.json").read_text())
    base_pred = json.loads((WORK / f"pred_{win}.json").read_text())
    merged = {"runs": base_pred["runs"], "variants": {BASE: base_pred["variants"][BASE]}}
    summary = {}
    for name, with_h2h in (("composite", False), ("composite+h2h", True)):
        dd = d + cp.h2h_coef * edge if with_h2h else d
        model = game_model.fit(dd, var, g1, g2, ot, po, src, gm_rows, weights)
        model_path = WORK / f"model_{win}_{name}.json"
        model_path.write_text(json.dumps(model.to_json()))
        ev2 = json.loads(json.dumps(events))
        n_edges = 0
        for ev in ev2["events"]:
            uid = ev["tournament_uid"]
            for p in ev["players"]:
                row = pre.loc[(uid, p["id"])]
                p["mu"], p["sd"] = float(row["mu"]), float(np.sqrt(row["var"]))
            if with_h2h:
                ev["h2h"] = event_edges(hist, [p["id"] for p in ev["players"]], int(first[uid]), cp)
                n_edges += len(ev["h2h"])
        ev_path = WORK / f"events_{win}_{name}.json"
        ev_path.write_text(json.dumps(ev2))
        part = WORK / f"pred_{win}_{name}.json"
        subprocess.run(["npx", "tsx", "scripts/backtest.ts", str(ev_path.resolve()), str(model_path.resolve()),
                        str(part.resolve()), str(args.runs), BASE], cwd="web", check=True)
        merged["variants"][f"{name}_form{FORM}"] = json.loads(part.read_text())["variants"][BASE]
        summary[name] = {"beta_rr": model.beta_rr, "edges": n_edges}
        print(name, summary[name], flush=True)
    tag = f"{win}p"
    shutil.copy(WORK / f"events_{win}.json", WORK / f"events_{tag}.json")
    (WORK / f"pred_{tag}.json").write_text(json.dumps(merged))
    res = score(tag)
    print(res["summary"].round(4).to_string())
    for k, v in res["tests"].items():
        print(k, v)
    rel = {v: {k: t.round(4).to_dict(orient="records") if hasattr(t, "round") else t for k, t in by.items()}
           for v, by in res["reliability"].items()}
    out = Path(f"reports/m8_tournaments_{win}.json")
    out.write_text(json.dumps({"window": win, "fit": summary, "variants": res["summary"].round(4).to_dict(orient="index"),
                               "tests": res["tests"], "reliability": rel}, indent=1, default=float))


if __name__ == "__main__":
    main()
