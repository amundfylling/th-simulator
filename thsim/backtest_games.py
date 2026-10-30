"""Game-level walk-forward backtest of the rating candidates.

Protocol: every rating system is run once over all games in date order, so each
game's ratings use only earlier games. Hyperparameters are tuned on 2015-2021,
models are compared on 2022-2024, and 2025-2026 stays untouched until the final
evaluation (``--final``).

Run: python -m thsim.backtest_games [--out reports/m2_ratings.md]
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from thsim import metrics
from thsim.data.loading import load_games_cached, load_players
from thsim.facts import outcome_codes, regulation_scores
from thsim.models import ordered_logit
from thsim.ratings import models as M
from thsim.ratings.tw_bt import TwBtParams

TUNE = ("2015-01-01", "2021-12-31")
SELECT = ("2022-01-01", "2024-12-31")
FINAL = ("2025-01-01", "2026-12-31")
SEASONED = 30

CANDIDATES = {
    "elo": M.EloParams(),
    "mov_elo": M.EloParams(mov=1.0),
    "glicko2": M.GlickoParams(),
    "kalman": M.KalmanParams(),
    "tw_bt": TwBtParams(),
}
SEARCH = {
    "elo": ["k", "k_new", "new_games"],
    "mov_elo": ["k", "k_new", "new_games", "mov"],
    "glicko2": ["tau", "phi0", "c_day", "phi_min"],
    "tw_bt": ["half_life_days", "ridge"],
    "kalman": ["obs_sd", "prior_sd", "prior_mean", "drift_sd_year", "drift_floor_day", "junior_mult", "gd_cap", "level_prior"],
}
# Variants with a prior mean of exactly 0 or a multiplier of 1 need additive steps.
ADDITIVE = {"prior_mean": 0.3, "level_prior": 0.1, "drift_floor_day": 0.005}


class Context:
    """Games, outcomes and evaluation masks shared by all candidates."""

    def __init__(self, games: pd.DataFrame, players: pd.DataFrame):
        self.games = games
        self.prep = M.prepare(games, players)
        self.n1, self.n2 = M.prior_counts(self.prep.a, self.prep.b)
        rg1, rg2 = regulation_scores(games)
        self.y3 = outcome_codes(rg1, rg2)
        self.p1_won = (games["g1"] > games["g2"]).to_numpy()
        played = ~games["walkover"].to_numpy()
        self.rr = played & games["stage_kind"].eq("rr").to_numpy()
        self.po = played & games["stage_kind"].eq("po").to_numpy() & (games["g1"] != games["g2"]).to_numpy()
        self.date = games["date"].to_numpy()
        self.seasoned = (self.n1 >= SEASONED) & (self.n2 >= SEASONED)
        self.known = (self.n1 >= 1) & (self.n2 >= 1)

    def window(self, span):
        lo, hi = np.datetime64(span[0]), np.datetime64(span[1])
        return (self.date >= lo) & (self.date <= hi)


def effective_gap(r: pd.DataFrame, beta: float | None) -> np.ndarray:
    d = (r["mu1"] - r["mu2"]).to_numpy()
    if beta is None:
        return d
    # Logistic-normal approximation: integrating the rating uncertainty flattens the curve.
    v = (r["var1"] + r["var2"]).to_numpy()
    return d / np.sqrt(1.0 + np.pi * beta**2 * v / 8.0)


def fit_and_score(ctx: Context, r: pd.DataFrame, fit_mask, eval_masks: dict, use_var: bool, max_fit: int | None = None):
    d = effective_gap(r, None)
    idx = np.flatnonzero(fit_mask)
    if max_fit and len(idx) > max_fit:
        idx = np.random.default_rng(0).choice(idx, max_fit, replace=False)
    beta, c = ordered_logit.fit(d[idx], ctx.y3[idx])
    if use_var and r["var1"].any():
        d = effective_gap(r, beta)
        beta, c = ordered_logit.fit(d[idx], ctx.y3[idx])
    # Playoff games: P(p1 wins the game) from a logistic on the same gap.
    po_idx = np.flatnonzero(ctx.po & ctx.window(TUNE) & ctx.known)
    if max_fit and len(po_idx) > max_fit:
        po_idx = np.random.default_rng(0).choice(po_idx, max_fit, replace=False)
    b_po = _fit_logistic(d[po_idx], ctx.p1_won[po_idx])
    out = {"beta": beta, "c": c, "beta_po": b_po}
    for name, mask in eval_masks.items():
        rr = mask & ctx.rr
        p = ordered_logit.probs(d[rr], beta, c)
        y = ctx.y3[rr]
        out[f"{name}_n"] = int(rr.sum())
        out[f"{name}_ll"] = metrics.log_loss(p, y)
        out[f"{name}_brier"] = metrics.brier(p, y)
        out[f"{name}_rps"] = metrics.rps(p, y)
        po = mask & ctx.po
        pp = 1.0 / (1.0 + np.exp(-b_po * d[po]))
        out[f"{name}_po_n"] = int(po.sum())
        out[f"{name}_po_ll"] = metrics.log_loss(pp, ctx.p1_won[po].astype(int))
    return out


def _fit_logistic(d, y):
    from scipy.optimize import minimize_scalar
    from scipy.special import log_expit

    s = max(np.std(d), 1e-9)
    z = d / s
    res = minimize_scalar(lambda b: -np.sum(np.where(y, log_expit(b * z), log_expit(-b * z))), bounds=(0, 50), method="bounded")
    return float(res.x / s)


def tune(ctx: Context, model: str, params, passes: int = 2, use_var: bool = True, log=print):
    fit_mask = ctx.rr & ctx.window(TUNE) & ctx.known
    masks = {"tune": ctx.window(TUNE) & ctx.known}

    def score(p):
        r = M.run(model, ctx.prep, p)
        return fit_and_score(ctx, r, fit_mask, masks, use_var, max_fit=250_000)["tune_ll"]

    best, best_ll = params, score(params)
    log(f"  {model} start {best_ll:.5f}")
    for _ in range(passes):
        improved = False
        for field in SEARCH[model]:
            value = getattr(best, field)
            if field in ADDITIVE or value == 0:
                step = ADDITIVE.get(field, 0.25)
                trials = [value - step, value + step]
            else:
                trials = [value * f for f in (0.6, 0.8, 1.25, 1.6)]
            for t in trials:
                if field != "prior_mean" and t < 0:
                    continue
                cand = replace(best, **{field: float(t)})
                ll = score(cand)
                if ll < best_ll - 1e-5:
                    best, best_ll, improved = cand, ll, True
                    log(f"  {model} {field}={t:.4g} -> {ll:.5f}")
        if not improved:
            break
    return best, best_ll


def evaluate(ctx: Context, model: str, params, use_var: bool, final: bool = False) -> dict:
    r = M.run(model, ctx.prep, params)
    fit_mask = ctx.rr & ctx.window(TUNE) & ctx.known
    masks = {
        "select_seasoned": ctx.window(SELECT) & ctx.seasoned,
        "select_all": ctx.window(SELECT) & ctx.known,
    }
    if final:
        masks |= {"final_seasoned": ctx.window(FINAL) & ctx.seasoned, "final_all": ctx.window(FINAL) & ctx.known}
    return fit_and_score(ctx, r, fit_mask, masks, use_var)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("reports/m2_ratings.md"))
    parser.add_argument("--params-out", type=Path, default=Path("reports/m2_params.json"))
    parser.add_argument("--final", action="store_true", help="also score the held-out 2025-2026 window")
    parser.add_argument("--models", nargs="*", default=list(CANDIDATES))
    parser.add_argument("--no-tune", action="store_true", help="reuse the tuned parameters in --params-out")
    args = parser.parse_args()
    t0 = time.time()
    ctx = Context(load_games_cached(), load_players())
    rows, tuned = [], (json.loads(args.params_out.read_text()) if args.params_out.exists() else {})
    for model in args.models:
        start = CANDIDATES[model]
        if model in tuned:
            # continue from the previous search instead of starting over
            start = type(start)(**tuned[model])
        if args.no_tune:
            params, tune_ll = start, float("nan")
        else:
            params, tune_ll = tune(ctx, model, start, passes={"tw_bt": 1, "kalman": 3}.get(model, 2))
        tuned[model] = asdict(params)
        has_var = model in ("glicko2", "kalman", "tw_bt")
        for use_var in ([False, True] if has_var else [False]):
            res = evaluate(ctx, model, params, use_var, final=args.final)
            res |= {"model": model + ("+var" if use_var else ""), "tune_ll": tune_ll}
            rows.append(res)
            print(res)
    df = pd.DataFrame(rows).set_index("model")
    args.params_out.write_text(json.dumps(tuned, indent=2))
    args.out.write_text(render(df, tuned, time.time() - t0, args.final))
    if args.no_tune:
        return print(args.out.read_text())
    print(args.out.read_text())


def render(df: pd.DataFrame, tuned: dict, seconds: float, final: bool) -> str:
    cols = ["tune_ll", "select_seasoned_ll", "select_seasoned_rps", "select_all_ll", "select_all_rps", "select_all_po_ll"]
    if final:
        cols += ["final_seasoned_ll", "final_all_ll", "final_all_po_ll"]
    t = df[cols].sort_values("select_all_ll")
    lines = ["| model | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for m, row in t.iterrows():
        lines.append(f"| {m} | " + " | ".join(f"{v:.4f}" for v in row) + " |")
    n = df.iloc[0]
    return f"""# Milestone 2: rating candidates, game-level walk-forward backtest

Ratings are computed walk-forward over all games (both sources, team events included, walkovers skipped).
Hyperparameters tuned on 2015-2021 round-robin 3-way log loss; models compared on 2022-2024
({int(n['select_all_n']):,} round-robin games with both players known, {int(n['select_seasoned_n']):,} with both seasoned 30+;
{int(n['select_all_po_n']):,} decided playoff games). `+var` = rating uncertainty integrated into the prediction.
Lower is better everywhere.

{chr(10).join(lines)}

Columns: `*_ll` 3-way log loss on round-robin games, `*_rps` ranked probability score, `*_po_ll` binary log loss
of the playoff game winner.

## Tuned parameters

```json
{json.dumps(tuned, indent=2)}
```

Runtime {seconds/60:.1f} min.
"""


if __name__ == "__main__":
    main()
