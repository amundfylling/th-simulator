"""Ten hypotheses for systematic prediction error, each tested walk-forward.

Every hypothesis adds pre-game information to the rating gap (or reshapes it). Its
coefficients are fitted on 2015-2021 together with the regulation model (ordered logit,
separate round-robin / playoff slope and draw cut, integrated over rating uncertainty)
and scored on 2022-2024 games. Ratings are the tuned Kalman filter's pre-game ratings,
so every feature uses only games played before the one being predicted.

Run: python -m thsim.research.hypotheses [--out reports/m7_hypotheses.md] [--final]
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numba as nb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from thsim.backtest_games import FINAL, SELECT, TUNE, Context
from thsim.data.loading import load_games_cached, load_players, load_tournaments
from thsim.models import ordered_logit
from thsim.ratings import models as M

GD_CLIP = 6.0


# --- regulation model with extra gap terms ------------------------------------------


def _probs(eta, c):
    pa = expit(eta - c)
    pb = expit(-eta - c)
    return pa, np.clip(1.0 - pa - pb, 1e-12, None), pb


class Fitted:
    def __init__(self, theta, k):
        self.theta = theta
        self.k = k

    def predict(self, d, sd, X, po):
        b_rr, c_rr, b_po, c_po = self.theta[:4]
        w = self.theta[4:]
        beta = np.where(po, b_po, b_rr)
        c = np.where(po, c_po, c_rr)
        shift = X @ w if self.k else 0.0
        out = np.zeros((len(d), 3))
        for x, wt in zip(ordered_logit.GH_X, ordered_logit.GH_W):
            pa, pd_, pb = _probs(beta * (d + sd * x + shift), c)
            out += wt * np.column_stack([pa, pd_, pb])
        return out


def fit(d, sd, X, po, y) -> Fitted:
    """Ordered logit on d + X @ w (w in goal units), slopes/cuts by game kind."""
    k = X.shape[1]
    rows = np.arange(len(y))

    def nll(theta):
        m = Fitted(theta, k)
        p = m.predict(d, sd, X, po)
        return -np.mean(np.log(np.clip(p[rows, y], 1e-12, 1)))

    x0 = np.r_[0.8, 0.37, 0.8, 0.37, np.zeros(k)]
    bounds = [(0.01, 5), (0, 3), (0.01, 5), (0, 3)] + [(-5, 5)] * k
    res = minimize(nll, x0, method="L-BFGS-B", bounds=bounds)
    return Fitted(res.x, k)


# --- walk-forward features -----------------------------------------------------------


@nb.njit(cache=True)
def _country_offsets(a, b, ca, cb, d, gd, skip, n_countries, lr):
    """Online pool offsets: learned only from cross-border games, before each game."""
    off = np.zeros(n_countries)
    o1 = np.zeros(len(a))
    o2 = np.zeros(len(a))
    for i in range(len(a)):
        if ca[i] < 0 or cb[i] < 0:
            continue
        o1[i] = off[ca[i]]
        o2[i] = off[cb[i]]
        if skip[i] or ca[i] == cb[i]:
            continue
        r = gd[i] - (d[i] + o1[i] - o2[i])
        off[ca[i]] += lr * r
        off[cb[i]] -= lr * r
    return o1, o2


@nb.njit(cache=True)
def _domestic_share(a, b, ca, cb, skip, n_players, prior):
    """Each player's share of earlier games against compatriots (shrunk toward 0.5)."""
    dom = np.zeros(n_players)
    tot = np.zeros(n_players)
    s1 = np.zeros(len(a))
    s2 = np.zeros(len(a))
    for i in range(len(a)):
        s1[i] = (dom[a[i]] + 0.5 * prior) / (tot[a[i]] + prior)
        s2[i] = (dom[b[i]] + 0.5 * prior) / (tot[b[i]] + prior)
        if skip[i] or ca[i] < 0 or cb[i] < 0:
            continue
        same = 1.0 if ca[i] == cb[i] else 0.0
        dom[a[i]] += same
        dom[b[i]] += same
        tot[a[i]] += 1
        tot[b[i]] += 1
    return s1, s2


def _event_running(a, b, t, res, skip):
    """Per game: each player's earlier games and summed residual in the same event."""
    n = len(a)
    played = (~skip).astype(float)
    long = pd.DataFrame({
        "t": np.r_[t, t], "p": np.r_[a, b], "order": np.r_[np.arange(n), np.arange(n)],
        "g": np.r_[played, played], "r": np.r_[res * played, -res * played],
    }).sort_values("order", kind="stable")
    grp = long.groupby(["t", "p"], sort=False)
    long["n_before"] = grp["g"].cumsum() - long["g"]
    long["r_before"] = grp["r"].cumsum() - long["r"]
    long = long.sort_index()
    return (long["n_before"].to_numpy()[:n], long["n_before"].to_numpy()[n:],
            long["r_before"].to_numpy()[:n], long["r_before"].to_numpy()[n:])


@nb.njit(cache=True)
def _days_since(a, b, day, skip, n_players):
    last = np.full(n_players, -1)
    g1 = np.zeros(len(a))
    g2 = np.zeros(len(a))
    for i in range(len(a)):
        g1[i] = day[i] - last[a[i]] if last[a[i]] >= 0 else -1
        g2[i] = day[i] - last[b[i]] if last[b[i]] >= 0 else -1
        if not skip[i]:
            last[a[i]] = day[i]
            last[b[i]] = day[i]
    return g1, g2


def series_state(games: pd.DataFrame) -> np.ndarray:
    """Playoff games: (p1 series wins - p2 series wins) before this game."""
    po = games["stage_kind"].eq("po").to_numpy()
    lo = np.minimum(games["p1"], games["p2"])
    hi = np.maximum(games["p1"], games["p2"])
    key = pd.Series(list(zip(games["stage_uid"], games["bracket"], games["round"], lo, hi)))
    lo_won = np.where(games["p1"].to_numpy() == lo.to_numpy(), games["g1"] > games["g2"], games["g2"] > games["g1"])
    hi_won = np.where(games["p1"].to_numpy() == hi.to_numpy(), games["g1"] > games["g2"], games["g2"] > games["g1"])
    df = pd.DataFrame({"key": key, "lo": lo_won.astype(int), "hi": hi_won.astype(int)})
    before = df.groupby("key")[["lo", "hi"]].cumsum() - df[["lo", "hi"]]
    diff_lo = (before["lo"] - before["hi"]).to_numpy()
    p1_is_lo = games["p1"].to_numpy() == lo.to_numpy()
    return np.where(po, np.where(p1_is_lo, diff_lo, -diff_lo), 0).astype(float)


class Features:
    def __init__(self, ctx: Context, games: pd.DataFrame, players: pd.DataFrame, params: M.KalmanParams):
        self.ctx = ctx
        prep = ctx.prep
        r = M.run("kalman", prep, params)
        self.mu1 = r["mu1"].to_numpy()
        self.mu2 = r["mu2"].to_numpy()
        self.d = self.mu1 - self.mu2
        self.sd = np.sqrt((r["var1"] + r["var2"]).to_numpy())
        self.var1 = r["var1"].to_numpy()
        self.var2 = r["var2"].to_numpy()
        self.gd = np.clip(prep.gd, -GD_CLIP, GD_CLIP)
        self.res = np.where(prep.skip, 0.0, self.gd - self.d)
        country = players.set_index("player_id")["country"]
        c1 = games["p1"].map(country)
        c2 = games["p2"].map(country)
        codes, self.countries = pd.factorize(pd.concat([c1, c2]))
        n = len(games)
        self.ca = codes[:n].astype(np.int64)
        self.cb = codes[n:].astype(np.int64)
        tours = load_tournaments().drop_duplicates("tournament_uid").set_index("tournament_uid")["country"]
        tc = games["tournament_uid"].map(tours)
        self.home1 = (c1.notna() & (c1 == tc)).to_numpy().astype(float)
        self.home2 = (c2.notna() & (c2 == tc)).to_numpy().astype(float)
        t = games["tournament_uid"].to_numpy().astype(np.int64)
        self.ev_n1, self.ev_n2, self.ev_r1, self.ev_r2 = _event_running(prep.a, prep.b, t, self.res, prep.skip)
        self.gap1, self.gap2 = _days_since(prep.a, prep.b, prep.day, prep.skip, len(prep.players))
        age = lambda bd: np.where(bd < 10**6, (prep.day - bd) / 365.25, np.nan)  # noqa: E731
        self.age1 = age(prep.birth_day[prep.a])
        self.age2 = age(prep.birth_day[prep.b])
        self.level = prep.level.astype(float)
        self.series = series_state(games)
        self.dom1, self.dom2 = _domestic_share(prep.a, prep.b, self.ca, self.cb, prep.skip, len(prep.players), 20.0)

    def country_offset(self, lr: float) -> np.ndarray:
        p = self.ctx.prep
        return _country_offsets(p.a, p.b, self.ca, self.cb, self.d, self.gd, p.skip, len(self.countries), lr)


# --- hypotheses --------------------------------------------------------------------------
# Each returns (d, sd, X): the gap fed to the model, its uncertainty and extra terms.


def h_baseline(F, **_):
    return F.d, F.sd, np.zeros((len(F.d), 0))


def h_country(F, lr=0.003, **_):
    o1, o2 = F.country_offset(lr)
    # Pool offset, and the same offset weighted by how insular each player is
    # (share of earlier games against compatriots): the fit decides which carries it.
    return F.d, F.sd, np.column_stack([o1 - o2, o1 * F.dom1 - o2 * F.dom2])


def h_gap_shape(F, **_):
    # Favourites over/under-confident at large gaps: a cubic term on the gap.
    return F.d, F.sd, np.column_stack([F.d * np.abs(F.d) / 4, F.d**3 / 16])


def h_margin_cap(F, cap=3.0, ctx=None, params=None, **_):
    r = M.run("kalman", ctx.prep, replace(params, gd_cap=cap))
    d = (r["mu1"] - r["mu2"]).to_numpy()
    sd = np.sqrt((r["var1"] + r["var2"]).to_numpy())
    return d, sd, np.zeros((len(d), 0))


def h_event_form(F, k=4.0, **_):
    f1 = F.ev_r1 / (F.ev_n1 + k)
    f2 = F.ev_r2 / (F.ev_n2 + k)
    return F.d, F.sd, (f1 - f2)[:, None]


def h_fatigue(F, **_):
    # Games already played in the event: tiredness late in a long day.
    return F.d, F.sd, ((F.ev_n1 - F.ev_n2) / 10)[:, None]


def h_age(F, **_):
    a1 = np.nan_to_num(F.age1, nan=30.0)
    a2 = np.nan_to_num(F.age2, nan=30.0)
    young = lambda a: np.clip(20 - a, 0, 10) / 5  # noqa: E731
    old = lambda a: np.clip(a - 45, 0, 30) / 10  # noqa: E731
    return F.d, F.sd, np.column_stack([young(a1) - young(a2), old(a1) - old(a2)])


def h_return(F, **_):
    # Back after a long break: rusty (mean shift), on top of the wider uncertainty.
    def rust(g):
        return np.where(g < 0, 0.0, np.clip((g - 90) / 365, 0, 3))

    return F.d, F.sd, (rust(F.gap1) - rust(F.gap2))[:, None]


def h_event_level(F, **_):
    # Bigger events: favourites more (or less) dominant. Levels 1-2 are the majors.
    major = ((F.level >= 1) & (F.level <= 2)).astype(float)
    minor = (F.level >= 5).astype(float)
    return F.d, F.sd, np.column_stack([F.d * major, F.d * minor])


def h_home(F, **_):
    return F.d, F.sd, (F.home1 - F.home2)[:, None]


def h_series(F, **_):
    # Momentum inside a playoff series: the score so far beyond the ratings.
    return F.d, F.sd, F.series[:, None]


def h_results_rating(F, ctx=None, **_):
    # Margins vs results: a rating that only sees win/draw/loss (Glicko-2) next to the
    # goal-difference rating. If big wins over weak players inflate a rating, it shows here.
    gp = json.loads(Path("reports/m2_params.json").read_text())["glicko2"]
    r = M.run("glicko2", ctx.prep, M.GlickoParams(**gp))
    g = (r["mu1"] - r["mu2"]).to_numpy() / 100.0
    return F.d, F.sd, (g - F.d)[:, None]


def h_combined(F, ctx=None, params=None, **_):
    """The hypotheses that helped on their own, together."""
    parts = [h_results_rating(F, ctx=ctx), h_country(F, lr=0.003), h_age(F), h_gap_shape(F)]
    return F.d, F.sd, np.column_stack([X for _, _, X in parts])


HYPOTHESES = {
    "baseline": (h_baseline, {}),
    "H1 country pools": (h_country, {"grid": {"lr": [0.001, 0.003, 0.01]}}),
    "H2 gap shape (fav over/under-confidence)": (h_gap_shape, {}),
    "H3 margin cap (blowouts inflate ratings)": (h_margin_cap, {"grid": {"cap": [2.0, 3.0, 4.0, 8.0]}}),
    "H4 form within the event": (h_event_form, {"grid": {"k": [2.0, 6.0]}}),
    "H5 fatigue over the event": (h_fatigue, {}),
    "H6 juniors improve, veterans decline": (h_age, {}),
    "H7 rust after a break": (h_return, {}),
    "H8 event level changes the gap": (h_event_level, {}),
    "H9 home country advantage": (h_home, {}),
    "H10 momentum inside a series": (h_series, {}),
    "H11 margins vs results (blowout specialists)": (h_results_rating, {}),
    "Combined: H11 + H1 + H6 + H2": (h_combined, {}),
}


def cluster_se(diff: np.ndarray, groups: np.ndarray) -> float:
    """Standard error of the mean, clustered by tournament."""
    s = pd.Series(diff - diff.mean()).groupby(groups).sum()
    return float(np.sqrt(len(s) / max(1, len(s) - 1) * np.sum(s**2)) / len(diff))


def run(final: bool) -> dict:
    games = load_games_cached()
    players = load_players()
    ctx = Context(games, players)
    params = M.KalmanParams(**json.loads(Path("reports/m2_params.json").read_text())["kalman"])
    F = Features(ctx, games, players, params)
    use = ctx.known & (ctx.rr | ctx.po)
    tune = use & ctx.window(TUNE)
    sel = use & ctx.window(SELECT)
    fin = use & ctx.window(FINAL)
    tour = games["tournament_uid"].to_numpy()
    rows = np.arange(len(games))
    y = ctx.y3
    po = ctx.po
    results = {}
    base_ll = {}
    for name, (fn, opts) in HYPOTHESES.items():
        t0 = time.time()
        grid = opts.get("grid", {})
        keys = list(grid)
        combos = [dict(zip(keys, v)) for v in np.array(np.meshgrid(*grid.values())).T.reshape(-1, len(keys))] if keys else [{}]
        best = None
        for kw in combos:
            d, sd, X = fn(F, ctx=ctx, params=params, **kw)
            m = fit(d[tune], sd[tune], X[tune], po[tune], y[tune])
            # Grid choice on the tuning window (in-sample), never on the scored windows.
            p = m.predict(d[tune], sd[tune], X[tune], po[tune])
            tune_ll = -np.mean(np.log(p[np.arange(tune.sum()), y[tune]]))
            if best is None or tune_ll < best[0]:
                best = (tune_ll, kw, m, d, sd, X)
        tune_ll, kw, m, d, sd, X = best
        out = {"params": {k: float(v) for k, v in kw.items()}, "coef": [float(x) for x in m.theta[4:]], "tune_ll": tune_ll}
        for wname, mask in (("select", sel), ("final", fin)) if final else (("select", sel),):
            p = m.predict(d[mask], sd[mask], X[mask], po[mask])
            ll = -np.log(p[np.arange(mask.sum()), y[mask]])
            if name == "baseline":
                base_ll[wname] = ll
            diff = ll - base_ll[wname]
            sub = {}
            for part, pm in (("all", np.ones(mask.sum(), bool)), ("po", po[mask]),
                             ("intl", (F.ca != F.cb)[mask] & (F.ca >= 0)[mask] & (F.cb >= 0)[mask])):
                sub[part] = {"n": int(pm.sum()), "ll": float(ll[pm].mean()), "delta_x1000": float(diff[pm].mean() * 1000),
                             "se_x1000": cluster_se(diff[pm], tour[mask][pm]) * 1000}
            out[wname] = sub
        results[name] = out
        print(name, out["params"], [round(c, 3) for c in out["coef"]], {k: round(v["delta_x1000"], 2) for k, v in out["select"].items()},
              f"{time.time() - t0:.0f}s", flush=True)
    return results


def report(res: dict, final: bool) -> str:
    lines = ["# Hypotheses for systematic prediction error", "",
             "Each hypothesis adds pre-game information to the Kalman rating gap; coefficients are fitted on",
             "2015-2021 together with the regulation model and scored on 2022-2024 (3-way regulation log loss,",
             "all games with both players known). Δ is ×1000 vs the baseline (negative = better), SE clustered",
             "by tournament. `intl` = players from different countries; `po` = playoff games.", "",
             "| hypothesis | chosen | coefficients (goals) | Δ all | ±SE | Δ intl | Δ po |", "|---|---|---|---|---|---|---|"]
    for name, r in sorted(res.items(), key=lambda kv: kv[1]["select"]["all"]["delta_x1000"]):
        s = r["select"]
        lines.append(f"| {name} | {r['params'] or ''} | {', '.join(f'{c:+.3f}' for c in r['coef'])} | "
                     f"{s['all']['delta_x1000']:+.2f} | {s['all']['se_x1000']:.2f} | {s['intl']['delta_x1000']:+.2f} | {s['po']['delta_x1000']:+.2f} |")
    b = res["baseline"]["select"]
    lines += ["", f"Baseline log loss: {b['all']['ll']:.4f} over {b['all']['n']:,} games ({b['intl']['n']:,} international, {b['po']['n']:,} playoff)."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("reports/m7_hypotheses.md"))
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    res = run(args.final)
    args.out.write_text(report(res, args.final))
    args.out.with_suffix(".json").write_text(json.dumps(res, indent=1))
    print(args.out.read_text())


if __name__ == "__main__":
    main()
