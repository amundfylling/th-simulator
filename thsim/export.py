"""Export everything the website needs: current player ratings and the fitted game model.

Run: python -m thsim.export [--out web/public/data]

Writes players.json (one row per active or ranked player), model.json (game model) and
meta.json (data freshness and validation summary).
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from thsim.data.loading import load_games, load_players, load_ranking
from thsim.models import game_model
from thsim.ratings import models as M

ACTIVE_YEARS = 6
FIT_YEARS = 8
HALF_LIFE_YEARS = 3.0


def load_settings(path: Path = Path("reports/m2_params.json")) -> tuple[M.KalmanParams, float]:
    params = M.KalmanParams(**json.loads(path.read_text())["kalman"]) if path.exists() else M.KalmanParams()
    form = Path("reports/m5_form.json")
    form_sd = json.loads(form.read_text())["form_sd"] if form.exists() else 0.0
    return params, form_sd


def build(out: Path, as_of: date | None = None) -> None:
    as_of = as_of or date.today()
    as_of_day = int(np.datetime64(as_of, "D").astype(np.int64))
    games = load_games()
    players = load_players()
    ranking = load_ranking()
    params, form_sd = load_settings()
    prep = M.prepare(games, players)

    # Game model: pre-game ratings, recency-weighted fit over the last FIT_YEARS.
    r = M.run("kalman", prep, params)
    d = (r["mu1"] - r["mu2"]).to_numpy()
    var = (r["var1"] + r["var2"]).to_numpy()
    n1, n2 = M.prior_counts(prep.a, prep.b)
    age = (as_of_day - prep.day) / 365.25
    fit_mask = (age <= FIT_YEARS) & (n1 >= 1) & (n2 >= 1) & ~prep.skip
    model = game_model.fit(
        d, var, games["g1"].to_numpy(), games["g2"].to_numpy(), games["ot"].to_numpy(),
        games["stage_kind"].eq("po").to_numpy(), games["source"].to_numpy(), fit_mask,
        0.5 ** (np.maximum(age, 0) / HALF_LIFE_YEARS),
    )
    model_json = model.to_json() | {"form_sd": form_sd, "rating_units": "goals"}

    # Current ratings.
    cur = M.kalman_final(prep, params, as_of_day)
    cur = cur.merge(players, on="player_id", how="left")
    # Players known only to bordshockey.net have no Scorpion record: take the name from games.
    game_names = pd.concat([
        games[["p1", "p1_name", "date"]].set_axis(["player_id", "game_name", "date"], axis=1),
        games[["p2", "p2_name", "date"]].set_axis(["player_id", "game_name", "date"], axis=1),
    ]).dropna().sort_values("date").drop_duplicates("player_id", keep="last")
    cur = cur.merge(game_names[["player_id", "game_name"]], on="player_id", how="left")
    cur["name"] = cur["name"].where(cur["name"].fillna("").ne(""), cur["game_name"])
    cur["last"] = pd.to_datetime(cur["last_day"], unit="D").dt.strftime("%Y-%m-%d")
    cur["active"] = (as_of_day - cur["last_day"]) <= ACTIVE_YEARS * 365.25
    if ranking is not None and len(ranking):
        cur = cur.merge(ranking[["ithf_id", "world_rank", "points", "nation"]], on="ithf_id", how="left")
    else:
        cur["world_rank"] = np.nan
        cur["points"] = np.nan
        cur["nation"] = None
    keep = cur["active"] | cur["world_rank"].notna()
    cur = cur.loc[keep & cur["name"].notna()].copy()
    cur = cur.sort_values(["world_rank", "mu"], ascending=[True, False], na_position="last")

    rows = [
        [
            int(p.player_id),
            p.name,
            (p.nation if isinstance(p.nation, str) else None) or (p.country if isinstance(p.country, str) else ""),
            None if pd.isna(p.ithf_id) else int(p.ithf_id),
            None if pd.isna(p.world_rank) else int(p.world_rank),
            None if pd.isna(p.points) else int(p.points),
            round(float(p.mu), 3),
            round(float(p.sd), 3),
            int(p.games),
            p.last,
        ]
        for p in cur.itertuples()
    ]
    out.mkdir(parents=True, exist_ok=True)
    (out / "players.json").write_text(json.dumps({
        "columns": ["id", "name", "country", "ithf_id", "world_rank", "ranking_points", "mu", "sd", "games", "last"],
        "rows": rows,
    }, ensure_ascii=False, separators=(",", ":")))
    (out / "model.json").write_text(json.dumps(model_json, separators=(",", ":")))
    meta = {
        "generated": as_of.isoformat(),
        "games": int(len(games)),
        "latest_game": str(games["date"].max().date()),
        "players": len(rows),
        "ranking_available": ranking is not None and len(ranking) > 0,
        "rating_params": params.__dict__,
        "game_model": {"beta_rr": model.beta_rr, "c_rr": model.c_rr, "beta_po": model.beta_po, "c_po": model.c_po,
                       "beta_ot": model.beta_ot},
    }
    for window in ("final", "select"):
        backtest = Path(f"reports/m5_tournaments_{window}.json")
        if backtest.exists():
            meta["tournament_backtest"] = json.loads(backtest.read_text())
            break
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print(f"Wrote {len(rows):,} players to {out}; model beta_rr={model.beta_rr:.3f} c={model.c_rr:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("web/public/data"))
    parser.add_argument("--as-of", type=date.fromisoformat, default=None)
    args = parser.parse_args()
    build(args.out, args.as_of)


if __name__ == "__main__":
    main()
