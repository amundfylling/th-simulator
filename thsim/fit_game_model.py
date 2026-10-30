"""Fit the game model on the chosen ratings, score it at game and series level, and run
shape checks (simulated vs real draw / OT / goal-difference shares per gap bin).

Run: python -m thsim.fit_game_model [--out reports/m3_game_model.md]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import comb, expit

from thsim import metrics
from thsim.backtest_games import SELECT, Context
from thsim.data.loading import load_games_cached, load_players
from thsim.models import game_model, ordered_logit
from thsim.ratings import models as M

FIT_SPAN = ("2015-01-01", "2021-12-31")  # fit on the tuning window, score on 2022-2024
GH_X, GH_W = np.polynomial.hermite_e.hermegauss(15)
GH_W = GH_W / GH_W.sum()


def load_params(path: Path = Path("reports/m2_params.json")) -> M.KalmanParams:
    if path.exists():
        return M.KalmanParams(**json.loads(path.read_text())["kalman"])
    return M.KalmanParams()


def series_prob(p: np.ndarray, need: np.ndarray) -> np.ndarray:
    """P(A wins a first-to-``need`` series) with per-game win probability p."""
    out = np.zeros_like(p)
    for n in np.unique(need):
        m = need == n
        q = 1.0 - p[m]
        out[m] = sum(comb(n - 1 + k, k) * p[m] ** n * q**k for k in range(n))
    return out


def integrate(f, d: np.ndarray, var: np.ndarray) -> np.ndarray:
    """E[f(d + sqrt(var) z)] with Gauss-Hermite quadrature (posterior rating draws)."""
    sd = np.sqrt(var)
    return sum(w * f(d + sd * x) for x, w in zip(GH_X, GH_W))


def game_win_prob(model: game_model.GameModel, d: np.ndarray) -> np.ndarray:
    p = model.reg_probs(d, playoff=True)
    return p[:, 0] + p[:, 1] * model.ot_prob(d)


def run(out: Path, model_out: Path | None, half_life: float = 3.0, fit_end: str = FIT_SPAN[1]) -> dict:
    games = load_games_cached()
    players = load_players()
    ctx = Context(games, players)
    params = load_params()
    r = M.run("kalman", ctx.prep, params)
    d = (r["mu1"] - r["mu2"]).to_numpy()
    var = (r["var1"] + r["var2"]).to_numpy()
    fit_mask = ctx.window((FIT_SPAN[0], fit_end)) & ctx.known & ~games["walkover"].to_numpy()
    playoff = games["stage_kind"].eq("po").to_numpy()
    g1 = games["g1"].to_numpy()
    g2 = games["g2"].to_numpy()
    ot = games["ot"].to_numpy()
    source = games["source"].to_numpy()

    # Likelihoods integrate over the rating posterior, so beta describes true skill gaps:
    # the simulator adds the posterior noise back by drawing ratings. Recent games weigh
    # more because scoring and draw rates drift over the years.
    age_years = (np.datetime64(fit_end) - games["date"].to_numpy()).astype("timedelta64[D]").astype(float) / 365.25
    weights = 0.5 ** (np.maximum(age_years, 0) / half_life)
    model = game_model.fit(d, var, g1, g2, ot, playoff, source, fit_mask, weights)

    sel = ctx.window(SELECT) & ctx.known & ~games["walkover"].to_numpy()
    res: dict = {"params": model.to_json()["regulation"] | {"overtime": model.beta_ot}}

    # Game level, scored by integrating over the rating posterior.
    rr = sel & ~playoff
    res["half_life"] = half_life
    p3 = integrate(lambda x: model.reg_probs(x), d[rr], var[rr])
    res["rr"] = _scores3(p3, ctx.y3[rr])
    # Draw cut by source (shared beta)
    p3src = np.empty_like(p3)
    for s in np.unique(source[rr]):
        m = source[rr] == s
        p3src[m] = integrate(lambda x: model.reg_probs(x, source=s), d[rr][m], var[rr][m])
    res["rr_by_source_c"] = _scores3(p3src, ctx.y3[rr])
    po = sel & playoff & (g1 != g2)
    pw = integrate(lambda x: game_win_prob(model, x), d[po], var[po])
    res["po_game"] = _scores2(pw, g1[po] > g2[po])
    po3 = integrate(lambda x: model.reg_probs(x, playoff=True), d[po], var[po])
    res["po_reg3"] = _scores3(po3, ctx.y3[po])
    otm = sel & playoff & ot
    pot = integrate(lambda x: model.ot_prob(x), d[otm], var[otm])
    res["ot_game"] = _scores2(pot, g1[otm] > g2[otm])
    res["ot_game_const"] = _scores2(np.full(otm.sum(), 0.5), g1[otm] > g2[otm])

    # Series level: per series length, ratings before the series' first game.
    res["series"] = series_scores(games, d, var, model, sel)

    # Shape checks with posterior draws.
    res["shape"] = game_model.shape_check(model, d[sel], var[sel], g1[sel], g2[sel], ot[sel], playoff[sel])

    out.write_text(render(res))
    if model_out:
        model_out.parent.mkdir(parents=True, exist_ok=True)
        model_out.write_text(json.dumps(model.to_json()))
    print(out.read_text())
    return res


def series_scores(games, d, var, model, mask) -> pd.DataFrame:
    po = games.loc[mask & games["stage_kind"].eq("po").to_numpy()].copy()
    po["d"] = d[po.index]
    po["var"] = var[po.index]
    po["a"] = np.minimum(po["p1"], po["p2"])
    sign = np.where(po["p1"].eq(po["a"]), 1.0, -1.0)
    po["d_a"] = po["d"] * sign
    po["a_win"] = np.where(po["p1"].eq(po["a"]), po["g1"] > po["g2"], po["g2"] > po["g1"])
    po["b_win"] = (po["g1"] != po["g2"]) & ~po["a_win"]
    po["round_key"] = po["round"].fillna(-1)
    po["pair_b"] = np.maximum(po["p1"], po["p2"])
    s = po.groupby(["tournament_uid", "stage_uid", "bracket", "stage_name", "round_key", "a", "pair_b"], sort=False).agg(
        d_a=("d_a", "first"), var=("var", "first"), a_wins=("a_win", "sum"), b_wins=("b_win", "sum"))
    s = s.loc[s["a_wins"] != s["b_wins"]]
    s["need"] = np.maximum(s["a_wins"], s["b_wins"])
    s = s.loc[s["need"].between(1, 4)]
    da = s["d_a"].to_numpy()
    need = s["need"].to_numpy()
    p = integrate(lambda x: series_prob(game_win_prob(model, x), need), da, s["var"].to_numpy())
    y = (s["a_wins"] > s["b_wins"]).to_numpy()
    rows = []
    for n in range(1, 5):
        m = need == n
        fav = np.where(da[m] >= 0, p[m], 1 - p[m])
        fav_won = np.where(da[m] >= 0, y[m], ~y[m])
        rows.append({"first_to": n, "series": int(m.sum()), "log_loss": metrics.log_loss(p[m], y[m].astype(int)),
                     "brier": metrics.brier(p[m], y[m].astype(float)), "mean_fav_prob": fav.mean(), "fav_won": fav_won.mean()})
    return pd.DataFrame(rows)


def _scores3(p, y):
    return {"n": len(y), "log_loss": metrics.log_loss(p, y), "brier": metrics.brier(p, y), "rps": metrics.rps(p, y),
            "pred_draw": float(p[:, 1].mean()), "real_draw": float((y == 1).mean())}


def _scores2(p, y):
    y = np.asarray(y)
    return {"n": len(y), "log_loss": metrics.log_loss(p, y.astype(int)), "brier": metrics.brier(p, y.astype(float)),
            "reliability": metrics.reliability(np.where(p >= 0.5, p, 1 - p), np.where(p >= 0.5, y, ~y))}


def render(res: dict) -> str:
    def row(name, s):
        return f"| {name} | {s['n']:,} | {s['log_loss']:.4f} | {s['brier']:.4f} | {s.get('rps', float('nan')):.4f} |"

    rel = lambda r: "\n".join(f"| {p:.3f} | {o:.3f} | {n:,} |" for p, o, n in r)  # noqa: E731
    shape = res["shape"].copy()
    shape["diff"] = shape["sim"] - shape["real"]
    shape_md = "\n".join(
        f"| {r.kind} | {r.stat} | {r.x_bin} | {r.real:.3f} | {r.sim:.3f} | {r.diff:+.3f} |" for r in shape.itertuples()
    )
    series = "\n".join(
        f"| {r.first_to} | {r.series:,} | {r.log_loss:.4f} | {r.brier:.4f} | {r.mean_fav_prob:.3f} | {r.fav_won:.3f} |"
        for r in res["series"].itertuples()
    )
    return f"""# Milestone 3: game model (regulation, overtime, scoreline)

Ratings: tuned Kalman goal-difference filter. Model fit on 2015-2021 games with both players known
(recency half-life {res['half_life']} years);
scores below are on 2022-2024 and integrate over each player's rating posterior.

Fitted parameters: `{json.dumps(res['params'])}`

## Game level

| set | games | log loss | Brier | RPS |
|---|---|---|---|---|
{row('round-robin 3-way', res['rr'])}
{row('round-robin 3-way, draw cut by source', res['rr_by_source_c'])}
{row('playoff regulation 3-way (win / OT / loss)', res['po_reg3'])}
{row('playoff game winner (incl. OT)', res['po_game'])}
{row('overtime winner, fitted beta_OT', res['ot_game'])}
{row('overtime winner, coin flip', res['ot_game_const'])}

Predicted vs real round-robin draw share: {res['rr']['pred_draw']:.3f} vs {res['rr']['real_draw']:.3f}.

### Reliability, playoff game winner (favourite's probability)

| forecast | observed | games |
|---|---|---|
{rel(res['po_game']['reliability'])}

### Reliability, overtime winner

| forecast | observed | games |
|---|---|---|
{rel(res['ot_game']['reliability'])}

## Series level (first to N, 2022-2024)

| first to | series | log loss | Brier | mean fav prob | fav won |
|---|---|---|---|---|---|
{series}

## Shape checks (simulated with posterior rating draws vs real), by x = beta * |gap|

| kind | stat | x bin | real | sim | sim - real |
|---|---|---|---|---|---|
{shape_md}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("reports/m3_game_model.md"))
    parser.add_argument("--model-out", type=Path, default=None)
    parser.add_argument("--half-life", type=float, default=3.0, help="recency half-life in years")
    parser.add_argument("--fit-end", default=FIT_SPAN[1], help="last date used for fitting (2024-12-31 for the final holdout)")
    args = parser.parse_args()
    run(args.out, args.model_out, args.half_life, args.fit_end)


if __name__ == "__main__":
    main()
