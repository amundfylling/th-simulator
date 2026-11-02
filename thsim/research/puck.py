"""What the h2h.puck.no ratings add to our predictions, tested walk-forward.

h2h.puck.no (https://h2h.puck.no/ratings) rates the same players with three ideas we did
not use:

* P1  Glicko-2 updated game by game on a soft margin score s = 1 / (1 + exp(-0.4 * gd)),
      with RD growing one volatility step per week of inactivity.
* P2  Its forecast shrinks the rating gap by Glicko's g(RD) factor before an ordered logit.
* P3  A head-to-head edge per pair: sum(result - expected) / (meetings + 20), added to the
      gap with weight 1.09. Some players match up badly against a specific opponent.

Protocol as in M2/M7: everything is fitted on 2015-2021 (ratings tuned, blend weight, edge
coefficient, regulation model), scored on 2022-2024, and 2025-2026 is scored with --final.
Metric: 3-way regulation log loss, Δ ×1000 vs today's Kalman model, SE clustered by tournament.

Run:
    python -m thsim.research.puck tune      # tune the Glicko-MoV rating on 2015-2021
    python -m thsim.research.puck fit       # blend weight + head-to-head coefficient -> reports/m8_params.json
    python -m thsim.research.puck report [--final]
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from thsim.backtest_games import FINAL, SELECT, TUNE, Context
from thsim.data.loading import load_games_cached, load_players
from thsim.models import ordered_logit
from thsim.ratings import composite as C
from thsim.ratings import models as M
from thsim.research import hypotheses as H

SCALE = C.SCALE
PUCK_DEFAULT = M.GlickoMovParams()  # the published h2h.puck.no settings
TUNED_PATH = Path("reports/m8_glicko_mov.json")


def kalman_params() -> M.KalmanParams:
    return M.KalmanParams(**json.loads(Path("reports/m2_params.json").read_text())["kalman"])


def tuned_glicko() -> M.GlickoMovParams:
    return M.GlickoMovParams(**json.loads(TUNED_PATH.read_text())["params"]) if TUNED_PATH.exists() else PUCK_DEFAULT


def _ll(m: H.Fitted, d, sd, X, po, y) -> np.ndarray:
    p = m.predict(d, sd, X, po)
    return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))


# --- tuning ---------------------------------------------------------------------------

TUNE_FIELDS = {"tau": "mul", "sigma0": "mul", "mov_k": "mul", "week_days": "mul", "phi0": "mul", "gd_cap": "mul",
               "prior_mean": 50.0}


def tune_glicko_mov(start: M.GlickoMovParams = PUCK_DEFAULT, passes: int = 4, sample: int = 200_000, log=print):
    """Coordinate search on 2015-2021 log loss of the rating alone (gap and RD integrated in the fit)."""
    games = load_games_cached()
    ctx = Context(games, load_players())
    mask = ctx.known & (ctx.rr | ctx.po) & ctx.window(TUNE)
    idx = np.flatnonzero(mask)
    idx = np.sort(np.random.default_rng(0).choice(idx, min(sample, len(idx)), replace=False))
    y, po = ctx.y3[idx], ctx.po[idx]
    zero = np.zeros((len(idx), 0))

    def score(p):
        r = M.run("glicko2_mov", ctx.prep, p)
        d = (r["mu1"] - r["mu2"]).to_numpy()[idx] / SCALE
        sd = np.sqrt((r["var1"] + r["var2"]).to_numpy()[idx]) / SCALE
        return float(np.mean(_ll(H.fit(d, sd, zero, po, y), d, sd, zero, po, y)))

    best, best_ll = start, score(start)
    start_ll = best_ll
    log(f"start {best_ll:.5f}")
    for _ in range(passes):
        improved = False
        for fld, kind in TUNE_FIELDS.items():
            v = getattr(best, fld)
            trials = [v * f for f in (0.6, 0.8, 1.25, 1.6)] if kind == "mul" else [v - kind, v + kind]
            for t in trials:
                cand = replace(best, **{fld: float(t)})
                ll = score(cand)
                if ll < best_ll - 1e-5:
                    best, best_ll, improved = cand, ll, True
                    log(f"  {fld}={t:.4g} -> {ll:.5f}")
        if not improved:
            break
    return best, {"start_ll": start_ll, "tuned_ll": best_ll}


# --- blend + head-to-head fit --------------------------------------------------------


def fit_composite(log=print) -> tuple[C.CompositeParams, dict]:
    games = load_games_cached()
    ctx = Context(games, load_players())
    kp = kalman_params()
    gp = tuned_glicko()
    tune = ctx.known & (ctx.rr | ctx.po) & ctx.window(TUNE)
    y, po = ctx.y3[tune], ctx.po[tune]
    zero = np.zeros((tune.sum(), 0))
    k = M.run("kalman", ctx.prep, kp)
    g = M.run("glicko2_mov", ctx.prep, gp)
    # Scale: match the Glicko gap's logit slope to the Kalman gap's (round-robin, tuning window).
    rr = np.flatnonzero(ctx.rr & ctx.window(TUNE) & ctx.known)[::3]
    b_k, _ = ordered_logit.fit((k["mu1"] - k["mu2"]).to_numpy()[rr], ctx.y3[rr])
    b_g, _ = ordered_logit.fit((g["mu1"] - g["mu2"]).to_numpy()[rr] / SCALE, ctx.y3[rr])
    scale = b_g / b_k
    log(f"scale {scale:.3f} goals per Glicko unit")
    grid = {}
    for w in (0.0, 0.5, 0.6, 0.7, 0.8, 1.0):
        for sm in (0.3, 0.4, 0.5, 0.65, 0.8, 1.0):
            cp = C.CompositeParams(glicko=gp, weight=w, scale=scale, sd_mult=sm)
            r = C.run(ctx.prep, kp, cp)
            d = (r["mu1"] - r["mu2"]).to_numpy()[tune]
            sd = np.sqrt((r["var1"] + r["var2"]).to_numpy()[tune])
            grid[f"{w}/{sm}"] = float(np.mean(_ll(H.fit(d, sd, zero, po, y), d, sd, zero, po, y)))
            log(f"  weight {w}, sd x{sm}: tune ll {grid[f'{w}/{sm}']:.5f}")
    w, sm = map(float, min(grid, key=grid.get).split("/"))
    cp = C.CompositeParams(glicko=gp, weight=w, scale=scale, sd_mult=sm)
    r = C.run(ctx.prep, kp, cp)
    d_all = (r["mu1"] - r["mu2"]).to_numpy()
    sd_all = np.sqrt((r["var1"] + r["var2"]).to_numpy())
    base = H.fit(d_all[tune], sd_all[tune], zero, po, y)
    cp.exp_beta, cp.exp_c = float(base.theta[0]), float(base.theta[1])
    resid = C.residuals(ctx.prep, r, cp)
    event = games["tournament_uid"].factorize()[0]
    h2h = {}
    for prior in (10.0, 20.0, 40.0):
        e, _ = C.h2h_edge(ctx.prep, resid, event, prior)
        X = e[tune][:, None]
        m = H.fit(d_all[tune], sd_all[tune], X, po, y)
        h2h[prior] = (float(np.mean(_ll(m, d_all[tune], sd_all[tune], X, po, y))), float(m.theta[4]))
        log(f"  h2h prior {prior}: tune ll {h2h[prior][0]:.5f}, coef {h2h[prior][1]:.3f}")
    prior = min(h2h, key=lambda p: h2h[p][0])
    cp.h2h_prior, cp.h2h_coef = prior, h2h[prior][1]
    return cp, {"weight_grid": grid, "h2h_grid": {str(p): v for p, v in h2h.items()}}


# --- report ---------------------------------------------------------------------------


def puck_gap(r) -> tuple[np.ndarray, np.ndarray]:
    """The puck.no forecast gap x = g(sqrt(RDa^2 + RDb^2)) * (ra - rb) / 173.7, and its RD-free twin."""
    raw = (r["mu1"] - r["mu2"]).to_numpy() / SCALE
    phi2 = (r["var1"] + r["var2"]).to_numpy() / SCALE**2
    return raw / np.sqrt(1.0 + 3.0 * phi2 / np.pi**2), raw


def evaluate(final: bool) -> dict:
    games = load_games_cached()
    players = load_players()
    ctx = Context(games, players)
    kp = kalman_params()
    F = H.Features(ctx, games, players, kp)
    cp = C.load()
    use = ctx.known & (ctx.rr | ctx.po)
    tune = use & ctx.window(TUNE)
    windows = {"select": use & ctx.window(SELECT)}
    if final:
        windows["final"] = use & ctx.window(FINAL)
    tour = games["tournament_uid"].to_numpy()
    y, po = ctx.y3, ctx.po
    n = len(games)
    zero = np.zeros((n, 0))
    nosd = np.zeros(n)
    event = games["tournament_uid"].factorize()[0]

    x_def, raw_def = puck_gap(M.run("glicko2_mov", ctx.prep, PUCK_DEFAULT))
    gt = M.run("glicko2_mov", ctx.prep, cp.glicko)
    x_tun, _ = puck_gap(gt)
    sd_tun = np.sqrt((gt["var1"] + gt["var2"]).to_numpy()) / SCALE
    r = C.run(ctx.prep, kp, cp)
    d_c = (r["mu1"] - r["mu2"]).to_numpy()
    sd_c = np.sqrt((r["var1"] + r["var2"]).to_numpy())
    resid = C.residuals(ctx.prep, r, cp)
    e_c, meet = C.h2h_edge(ctx.prep, resid, event, cp.h2h_prior)
    # Edge on today's Kalman model alone (expected score from its own regulation fit).
    base = H.fit(F.d[tune], F.sd[tune], zero[tune], po[tune], y[tune])
    pk = ordered_logit.integrated_probs(F.d, F.sd**2, base.theta[0], base.theta[1])
    e_k, _ = C.h2h_edge(ctx.prep, ctx.prep.score - (pk[:, 0] + 0.5 * pk[:, 1]), event, cp.h2h_prior)
    gr = M.run("glicko2", ctx.prep, M.GlickoParams(**json.loads(Path("reports/m2_params.json").read_text())["glicko2"]))
    g_res = (gr["mu1"] - gr["mu2"]).to_numpy() / 100.0
    cands = {
        "today: Kalman": (F.d, F.sd, zero),
        "P1 h2h.puck.no rating, published settings, raw gap": (raw_def, nosd, zero),
        "P1+P2 h2h.puck.no rating + forecast, published settings": (x_def, nosd, zero),
        "P1+P2 same, tuned on our 2015-2021 games": (x_tun, nosd, zero),
        "P1 tuned, rating uncertainty integrated instead of g(RD)": (puck_gap(gt)[1], sd_tun, zero),
        "P3 head-to-head edge on today's Kalman": (F.d, F.sd, e_k[:, None]),
        "H11 (M7) Kalman + results-only Glicko, for reference": (F.d, F.sd, (g_res - F.d)[:, None]),
        f"Composite rating (weight {cp.weight:g} on tuned Glicko-MoV, sd ×{cp.sd_mult:g})": (d_c, sd_c, zero),
        "Composite + head-to-head edge (shipped)": (d_c, sd_c, e_c[:, None]),
    }
    res, base_ll = {}, {}
    for name, (d, sd, X) in cands.items():
        t0 = time.time()
        if name.startswith("Composite + head"):
            # Fixed coefficient from the fit (the edge is already in goal units after scaling).
            X = X * cp.h2h_coef
            m = H.fit(d[tune] + X[tune, 0], sd[tune], zero[tune], po[tune], y[tune])
            d, X = d + X[:, 0], zero
        else:
            m = H.fit(d[tune], sd[tune], X[tune], po[tune], y[tune])
        out = {"coef": [float(c) for c in m.theta[4:]], "theta": [float(c) for c in m.theta[:4]]}
        for wname, mask in windows.items():
            ll = _ll(m, d[mask], sd[mask], X[mask], po[mask], y[mask])
            base_ll.setdefault(wname, ll)
            diff = ll - base_ll[wname]
            sub = {}
            for part, pm in (("all", np.ones(mask.sum(), bool)), ("po", po[mask]), ("met5", meet[mask] >= 5)):
                sub[part] = {"n": int(pm.sum()), "ll": float(ll[pm].mean()), "delta_x1000": float(diff[pm].mean() * 1000),
                             "se_x1000": H.cluster_se(diff[pm], tour[mask][pm]) * 1000}
            out[wname] = sub
        res[name] = out
        print(name, {w: round(out[w]["all"]["delta_x1000"], 2) for w in windows}, f"{time.time() - t0:.0f}s", flush=True)
    return res


def report(res: dict, final: bool) -> str:
    cp = C.load()
    tuned = json.loads(TUNED_PATH.read_text())
    fit = json.loads(C.PARAMS_PATH.read_text())
    wins = ["select", "final"] if final else ["select"]
    head = "| candidate | " + " | ".join(f"{w} log loss | Δ ×1000 (±SE) | Δ pairs met 5+ | Δ playoff" for w in wins) + " |"
    lines = ["# What the h2h.puck.no ratings add (game level)", "",
             "Ideas from https://h2h.puck.no/ratings, scored on our data like the M7 hypotheses. Everything is",
             "fitted on 2015-2021 (rating settings, blend weight, head-to-head coefficient, regulation model);",
             "`select` = 2022-2024, `final` = 2025-2026 (untouched until the end). 3-way regulation log loss, Δ vs",
             "today's Kalman model (negative = better), SE clustered by tournament. `pairs met 5+` = games where",
             "the two players had met at least 5 times before.", "",
             head, "|---" * (1 + 4 * len(wins)) + "|"]
    for name, r in res.items():
        cells = []
        for w in wins:
            s = r[w]
            cells += [f"{s['all']['ll']:.4f}", f"{s['all']['delta_x1000']:+.2f} ({s['all']['se_x1000']:.2f})",
                      f"{s['met5']['delta_x1000']:+.2f}", f"{s['po']['delta_x1000']:+.2f}"]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    b = res["today: Kalman"]
    lines += ["", "Games: " + ", ".join(f"{w} {b[w]['all']['n']:,} ({b[w]['met5']['n']:,} pairs met 5+, {b[w]['po']['n']:,} playoff)" for w in wins) + ".",
              "", "For scale: switching from Elo to the Kalman ratings was worth about −13 on this metric; the best M7",
              "hypothesis (H11) about −1.4.", "",
              "## Fitted settings", "",
              f"Glicko-MoV tuned on 2015-2021 (tuning log loss {tuned['start_ll']:.4f} with the published settings → "
              f"{tuned['tuned_ll']:.4f}):", "", "```json", json.dumps(asdict(cp.glicko), indent=1), "```", "",
              f"Blend weight on the Glicko-MoV rating: **{cp.weight:g}** (scale {cp.scale:.3f} goals per 173.7 rating points). "
              f"Blended sd multiplier: **{cp.sd_mult:g}** (Glicko's RD overstates the spread). "
              "Tuning log loss by weight/sd multiplier: " + ", ".join(f"{k}: {v:.5f}" for k, v in fit["fit"]["weight_grid"].items()) + ".", "",
              f"Head-to-head edge: prior {cp.h2h_prior:g} meetings, coefficient {cp.h2h_coef:.3f} goals per unit of edge. "
              "Tuning log loss by prior: " + ", ".join(f"{k}: {v[0]:.5f}" for k, v in fit["fit"]["h2h_grid"].items()) + "."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["tune", "fit", "report"])
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("reports/m8_puck.md"))
    args = parser.parse_args()
    if args.action == "tune":
        start = tuned_glicko()
        best, info = tune_glicko_mov(start)
        if TUNED_PATH.exists():
            info["start_ll"] = json.loads(TUNED_PATH.read_text())["start_ll"]
        TUNED_PATH.write_text(json.dumps({"params": asdict(best), **info}, indent=1))
        print(TUNED_PATH.read_text())
    elif args.action == "fit":
        cp, info = fit_composite()
        C.PARAMS_PATH.write_text(json.dumps({"composite": cp.to_json(), "fit": info}, indent=1))
        print(C.PARAMS_PATH.read_text())
    else:
        res = evaluate(args.final)
        args.out.write_text(report(res, args.final))
        args.out.with_suffix(".json").write_text(json.dumps(res, indent=1))
        print(args.out.read_text())


if __name__ == "__main__":
    main()
