"""M9: ideas from a LightGBM experiment, tested on our data like M7/M8.

Part 1, cheap terms on top of today's model (composite rating + head-to-head edge):
  * per-player draw rate moves the draw cut,
  * extra uncertainty for players with few games,
  * recent form (EWMA of goal residuals across events).
Part 2, a LightGBM layer: 3-way regulation outcome from ~30 pre-game features, trained on
2015-2021 with both orientations of every game (order-free), scored like everything else.

Run: python -m thsim.research.m9 [--final]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from thsim.backtest_games import FINAL, SELECT, TUNE, Context
from thsim.data.loading import load_games_cached, load_players
from thsim.models import gbm as G
from thsim.models import ordered_logit
from thsim.ratings import composite as C
from thsim.research import features as FT
from thsim.research import hypotheses as H
from thsim.research.puck import kalman_params

GBM_FEATURES = G.FEATURES


# --- extended ordered logit: gap terms, draw-cut terms, uncertainty terms -----------


class Ext:
    def __init__(self, theta, kx, kz, ks):
        self.theta, self.kx, self.kz, self.ks = theta, kx, kz, ks

    def predict(self, d, sd, po, X, Z, S):
        t = self.theta
        b = np.where(po, t[2], t[0])
        c = np.where(po, t[3], t[1])
        i = 4
        if self.kx:
            d = d + X @ t[i:i + self.kx]
        i += self.kx
        if self.kz:
            c = np.maximum(c + Z @ t[i:i + self.kz], 1e-3)
        i += self.kz
        if self.ks:
            sd = sd * np.exp(S @ t[i:i + self.ks])
        out = np.zeros((len(d), 3))
        for x, w in zip(ordered_logit.GH_X, ordered_logit.GH_W):
            eta = b * (d + sd * x)
            pa, pb = expit(eta - c), expit(-eta - c)
            out += w * np.column_stack([pa, np.clip(1 - pa - pb, 1e-12, None), pb])
        return out


def fit_ext(d, sd, po, y, X, Z, S) -> Ext:
    kx, kz, ks = X.shape[1], Z.shape[1], S.shape[1]
    rows = np.arange(len(y))

    def nll(theta):
        p = Ext(theta, kx, kz, ks).predict(d, sd, po, X, Z, S)
        return -np.mean(np.log(np.clip(p[rows, y], 1e-12, 1)))

    x0 = np.r_[0.85, 0.4, 0.8, 0.35, np.zeros(kx + kz + ks)]
    bounds = [(0.01, 5), (0, 3), (0.01, 5), (0, 3)] + [(-5, 5)] * (kx + kz + ks)
    return Ext(minimize(nll, x0, method="L-BFGS-B", bounds=bounds).x, kx, kz, ks)


def cheap_terms(f: pd.DataFrame) -> dict[str, tuple]:
    n = len(f)
    e = np.zeros((n, 0))
    dr = (f["draw_rate_1"] + f["draw_rate_2"] - 0.25).to_numpy()[:, None]
    newc = (1 / np.sqrt(np.expm1(f["games_1"]) + 1) + 1 / np.sqrt(np.expm1(f["games_2"]) + 1)).to_numpy()[:, None]
    form = (f["form_1"] - f["form_2"]).to_numpy()[:, None]
    return {
        "today (composite + head-to-head)": (e, e, e),
        "+ per-player draw rate (draw cut)": (e, -dr * 4, e),
        "+ newcomer uncertainty": (e, e, newc),
        "+ recent form (EWMA residual, 30 games)": (form, e, e),
        "+ all three": (form, -dr * 4, newc),
    }


# --- LightGBM (thsim.models.gbm) ----------------------------------------------------


def gbm_fit(f: pd.DataFrame, y: np.ndarray, train: np.ndarray, valid: np.ndarray, log=print):
    return G.train(f, y, train, valid, log=log)


def gbm_predict(model, f: pd.DataFrame) -> np.ndarray:
    return G.predict(model, f)


# --- run -----------------------------------------------------------------------------


def run(final: bool) -> dict:
    games = load_games_cached()
    ctx = Context(games, load_players())
    kp, cp = kalman_params(), C.load()
    t0 = time.time()
    f = FT.build(ctx, games, kp, cp)
    print(f"features {time.time() - t0:.0f}s", flush=True)
    use = ctx.known & (ctx.rr | ctx.po)
    tune = use & ctx.window(TUNE)
    windows = {"select": use & ctx.window(SELECT)}
    if final:
        windows["final"] = use & ctx.window(FINAL)
    y, po = ctx.y3, ctx.po
    tour = games["tournament_uid"].to_numpy()
    g1, g2 = games["g1"].to_numpy(), games["g2"].to_numpy()
    d = (f["d"] + f["h2h"]).to_numpy()
    sd = f["sd"].to_numpy()
    preds: dict[str, np.ndarray] = {}
    coefs = {}
    for name, (X, Z, S) in cheap_terms(f).items():
        m = fit_ext(d[tune], sd[tune], po[tune], y[tune], X[tune], Z[tune], S[tune])
        preds[name] = m.predict(d, sd, po, X, Z, S)
        coefs[name] = [float(t) for t in m.theta]
        print(name, np.round(m.theta, 3), flush=True)
    # LightGBM: train 2015-2020, early-stop on 2021 (both inside the tuning window).
    train = tune & (ctx.date < np.datetime64("2021-01-01"))
    valid = tune & (ctx.date >= np.datetime64("2021-01-01"))
    model = gbm_fit(f, y, train, valid)
    preds["LightGBM layer (ratings + 30 features)"] = gbm_predict(model, f)
    imp = pd.Series(model.feature_importance("gain"), index=GBM_FEATURES).sort_values(ascending=False)
    res = {"coefs": coefs, "importance": (imp / imp.sum()).round(4).head(20).to_dict(), "gbm_rounds": model.best_iteration}
    base_name = "today (composite + head-to-head)"
    for wname, mask in windows.items():
        idx = np.flatnonzero(mask)
        base = -np.log(np.clip(preds[base_name][idx, y[idx]], 1e-12, 1))
        dec = g1[idx] != g2[idx]
        few = (np.expm1(f["games_1"].to_numpy()[idx]) < 30) | (np.expm1(f["games_2"].to_numpy()[idx]) < 30)
        out = {}
        for name, p in preds.items():
            pp = p[idx]
            ll = -np.log(np.clip(pp[np.arange(len(idx)), y[idx]], 1e-12, 1))
            diff = ll - base
            # Favourite = the side with the larger win probability; decided games only.
            fav_ok = ((pp[:, 0] > pp[:, 2]) == (g1[idx] > g2[idx]))[dec]
            out[name] = {
                "ll": float(ll.mean()), "delta_x1000": float(diff.mean() * 1000),
                "se_x1000": H.cluster_se(diff, tour[idx]) * 1000,
                "delta_po_x1000": float(diff[po[idx]].mean() * 1000),
                "delta_few_x1000": float(diff[few].mean() * 1000),
                "fav_won": float(fav_ok.mean()), "top_pick": float(np.mean(pp.argmax(1) == y[idx])),
                "pred_draw": float(pp[~po[idx], 1].mean()), "real_draw": float(np.mean(y[idx][~po[idx]] == 1)),
            }
        res[wname] = out
        res[f"{wname}_n"] = {"games": int(len(idx)), "few": int(few.sum()), "po": int(po[idx].sum())}
        for name, o in out.items():
            print(wname, name, {k: round(v, 4) for k, v in o.items()}, flush=True)
    model.save_model("data_cache/m9_gbm.txt")
    return res


def report(res: dict, final: bool) -> str:
    wins = ["select", "final"] if final else ["select"]
    lines = ["# M9: ideas from a LightGBM experiment (game level)", "",
             "Baseline is today's shipped model (composite rating + head-to-head edge, M8). Everything is fitted on",
             "2015-2021 (LightGBM: trained 2015-2020, early-stopped on 2021); `select` = 2022-2024, `final` = 2025-2026.",
             "Log loss is 3-way regulation (win / draw-or-OT / loss), Δ ×1000 vs today (negative = better), SE clustered",
             "by tournament. `few` = games where a player had fewer than 30 earlier games. `fav won` = share of decided",
             "games won by the side the model favoured; `top pick` = most likely of the three outcomes happened.", "",
             "LightGBM is order-free: every training game appears from both sides, and predictions average both views,",
             "so player 1 / player 2 order cannot leak the result.", ""]
    for w in wins:
        n = res[f"{w}_n"]
        lines += [f"## {w} ({n['games']:,} games, {n['few']:,} with a newcomer, {n['po']:,} playoff)", "",
                  "| model | log loss | Δ ×1000 (±SE) | Δ playoff | Δ few | fav won | top pick | draws pred / real (rr) |",
                  "|---|---|---|---|---|---|---|---|"]
        for name, o in res[w].items():
            lines.append(f"| {name} | {o['ll']:.4f} | {o['delta_x1000']:+.2f} ({o['se_x1000']:.2f}) | {o['delta_po_x1000']:+.2f} | "
                         f"{o['delta_few_x1000']:+.2f} | {o['fav_won']:.2%} | {o['top_pick']:.2%} | {o['pred_draw']:.3f} / {o['real_draw']:.3f} |")
        lines.append("")
    lines += ["## LightGBM feature importance (share of gain, top 20)", "", "| feature | share |", "|---|---|"]
    lines += [f"| {k} | {v:.3f} |" for k, v in res["importance"].items()]
    lines += ["", f"Boosting rounds: {res['gbm_rounds']}."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("reports/m9_features.md"))
    args = parser.parse_args()
    res = run(args.final)
    args.out.write_text(report(res, args.final))
    args.out.with_suffix(".json").write_text(json.dumps(res, indent=1))
    print(args.out.read_text())


if __name__ == "__main__":
    main()
