"""Tournament-level check of the rating adjustments that helped at game level.

Each candidate turns into per-player ratings (Kalman mean plus fitted adjustments), a
game model refitted on those ratings, and a full run of the website engine over the
reconstructed events of the M5 backtest. Coefficients and game model use games up to
2021 for the 2022-2024 window and up to 2024 for the 2025-2026 window.

Run (after python -m thsim.backtest_tournaments build/simulate for the window):
    python -m thsim.research.tournament_check --window select
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from thsim.backtest_games import Context
from thsim.backtest_tournaments import WORK, _pre_event_ratings, score
from thsim.data.loading import load_games_cached, load_players
from thsim.fit_game_model import load_params
from thsim.models import game_model
from thsim.ratings import models as M
from thsim.research import hypotheses as H

FIT_END = {"select": "2021-12-31", "final": "2024-12-31"}
FORM = 0.45


def player_terms(F: H.Features, ctx: Context) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """Per-player adjustment terms (p1 side, p2 side) for each candidate component."""
    gp = json.loads(Path("reports/m2_params.json").read_text())["glicko2"]
    g = M.run("glicko2", ctx.prep, M.GlickoParams(**gp))
    # Offsets frozen at each event's start: every entrant's pre-event rating sees no result of the event.
    o1, o2 = F.country_offset(0.003, freeze_by_event=True)
    young = lambda a: np.clip(20 - np.nan_to_num(a, nan=30.0), 0, 10) / 5  # noqa: E731
    old = lambda a: np.clip(np.nan_to_num(a, nan=30.0) - 45, 0, 30) / 10  # noqa: E731
    return {
        "results": [(g["mu1"].to_numpy() / 100 - F.mu1, g["mu2"].to_numpy() / 100 - F.mu2)],
        "country": [(o1, o2), (o1 * F.dom1, o2 * F.dom2)],
        "age": [(young(F.age1), young(F.age2)), (old(F.age1), old(F.age2))],
    }


CANDIDATES = {"results": ["results"], "results+country+age": ["results", "country", "age"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", default="select", choices=list(FIT_END))
    parser.add_argument("--runs", type=int, default=2000)
    args = parser.parse_args()
    win = args.window
    games = load_games_cached()
    players = load_players()
    ctx = Context(games, players)
    params = load_params()
    F = H.Features(ctx, games, players, params)
    terms = player_terms(F, ctx)
    fit_span = ("2015-01-01", FIT_END[win])
    fit_rows = ctx.known & (ctx.rr | ctx.po) & ctx.window(fit_span)
    gm_rows = ctx.window(fit_span) & ctx.known & ~games["walkover"].to_numpy()
    age_years = (np.datetime64(FIT_END[win]) - games["date"].to_numpy()).astype("timedelta64[D]").astype(float) / 365.25
    weights = 0.5 ** (np.maximum(age_years, 0) / 3.0)
    events = json.loads((WORK / f"events_{win}.json").read_text())
    base_pred = json.loads((WORK / f"pred_{win}.json").read_text())
    merged = {"runs": base_pred["runs"], "variants": {"kalman_form0.45": base_pred["variants"][f"kalman_form{FORM}"]}}
    summary = {}
    for name, parts in CANDIDATES.items():
        cols = [t for p in parts for t in terms[p]]
        X = np.column_stack([t1 - t2 for t1, t2 in cols])
        fitted = H.fit(F.d[fit_rows], F.sd[fit_rows], X[fit_rows], ctx.po[fit_rows], ctx.y3[fit_rows])
        w = fitted.theta[4:]
        mu1 = F.mu1 + sum(wi * t1 for wi, (t1, _) in zip(w, cols))
        mu2 = F.mu2 + sum(wi * t2 for wi, (_, t2) in zip(w, cols))
        d = mu1 - mu2
        var = F.var1 + F.var2
        model = game_model.fit(d, var, games["g1"].to_numpy(), games["g2"].to_numpy(), games["ot"].to_numpy(),
                               games["stage_kind"].eq("po").to_numpy(), games["source"].to_numpy(), gm_rows, weights)
        model_path = WORK / f"model_{win}_{name}.json"
        model_path.write_text(json.dumps(model.to_json()))
        pre = _pre_event_ratings(games, {"mu": (mu1, mu2)})
        ev2 = json.loads(json.dumps(events))
        for ev in ev2["events"]:
            for p in ev["players"]:
                p["mu"] = float(pre.loc[(ev["tournament_uid"], p["id"]), "mu"])
        ev_path = WORK / f"events_{win}_{name}.json"
        ev_path.write_text(json.dumps(ev2))
        part = WORK / f"pred_{win}_{name}.json"
        subprocess.run(["npx", "tsx", "scripts/backtest.ts", str(ev_path.resolve()), str(model_path.resolve()),
                        str(part.resolve()), str(args.runs), f"kalman_form{FORM}"], cwd="web", check=True)
        merged["variants"][f"{name}_form{FORM}"] = json.loads(part.read_text())["variants"][f"kalman_form{FORM}"]
        summary[name] = {"coef": [float(x) for x in w], "beta_rr": model.beta_rr}
        print(name, summary[name], flush=True)
    tag = f"{win}x"
    shutil.copy(WORK / f"events_{win}.json", WORK / f"events_{tag}.json")
    (WORK / f"pred_{tag}.json").write_text(json.dumps(merged))
    res = score(tag)
    print(res["summary"].round(4).to_string())
    for k, v in res.get("tests", {}).items():
        print(k, v)
    out = Path(f"reports/m7_tournaments_{win}.json")
    out.write_text(json.dumps({"coef": summary, "variants": res["summary"].round(4).to_dict(orient="index"),
                               "tests": {k: v for k, v in res.get("tests", {}).items()}}, indent=1, default=float))


if __name__ == "__main__":
    main()
