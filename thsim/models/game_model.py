"""Game model: regulation outcome, overtime and scoreline, all driven by a rating gap.

* Regulation: ordered logit, P(A) = sigma(beta*d - c), P(B) = sigma(-beta*d - c).
  Round-robin and playoff games get their own (beta, c); c may differ by source.
* Overtime (a winner is needed): A wins with sigma(beta_ot * d); +1 goal to the winner.
* Scoreline: drawn from real regulation scorelines with the same outcome and a similar
  gap. Gaps are binned on x = beta*|d| (logit units, so the table does not depend on the
  rating scale) and smoothed across bins with a Gaussian kernel.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, log_expit

from thsim.models import ordered_logit

X_BIN = 0.25
N_BINS = 24  # x in [0, 6)
KERNEL_BW = 0.25
MAX_GOALS = 15
OUTCOMES = ("fav", "draw", "dog")


@dataclass
class GameModel:
    beta_rr: float
    c_rr: float
    beta_po: float
    c_po: float
    beta_ot: float
    c_by_source: dict = field(default_factory=dict)
    # scorelines[kind][outcome] -> array (N_BINS, MAX_GOALS+1, MAX_GOALS+1) of probabilities
    # over (favourite goals, underdog goals), conditioned on the outcome; kind "rr" or "po".
    scorelines: dict = field(default_factory=dict)

    def reg_probs(self, d: np.ndarray, playoff: bool = False, source: str | None = None) -> np.ndarray:
        beta, c = (self.beta_po, self.c_po) if playoff else (self.beta_rr, self.c_by_source.get(source, self.c_rr))
        return ordered_logit.probs(np.asarray(d, float), beta, c)

    def ot_prob(self, d: np.ndarray) -> np.ndarray:
        return expit(self.beta_ot * np.asarray(d, float))

    def to_json(self) -> dict:
        tables = {}
        for kind, by_outcome in self.scorelines.items():
            tables[kind] = {}
            for o in OUTCOMES:
                t = by_outcome[o]
                tables[kind][o] = [
                    [[int(i), int(j), round(float(t[b, i, j]), 6)] for i, j in zip(*np.nonzero(t[b] > 1e-5))]
                    for b in range(N_BINS)
                ]
        return {
            "regulation": {"rr": {"beta": self.beta_rr, "c": self.c_rr, "c_by_source": self.c_by_source},
                           "po": {"beta": self.beta_po, "c": self.c_po}},
            "overtime": {"beta": self.beta_ot},
            "scoreline": {"x_bin": X_BIN, "n_bins": N_BINS, "beta": self.beta_rr, "tables": tables},
        }


def _fit_slope(d: np.ndarray, y: np.ndarray, weights: np.ndarray, var: np.ndarray) -> float:
    """Logistic slope through the origin, integrated over the gap posterior."""
    s = max(np.std(d), 1e-9)
    z = d / s
    sd = np.sqrt(var) / s
    w = weights / weights.mean()

    def nll(b):
        p = sum(wt * expit(b * (z + sd * x)) for x, wt in zip(ordered_logit.GH_X, ordered_logit.GH_W))
        return -np.sum(w * np.log(np.clip(np.where(y, p, 1 - p), 1e-12, 1)))

    return float(minimize_scalar(nll, bounds=(0, 50), method="bounded").x / s)


def _fit_c(d, y, beta, weights, var):
    """Refit only the draw cut for a subset, holding beta fixed."""
    rows = np.arange(len(y))

    def nll(c):
        p = ordered_logit.integrated_probs(d, var, beta, c)
        return -np.sum(weights * np.log(np.clip(p[rows, y], 1e-12, 1)))

    return float(minimize_scalar(nll, bounds=(0, 3), method="bounded").x)


def scoreline_tables(x: np.ndarray, fav_goals: np.ndarray, dog_goals: np.ndarray, weights: np.ndarray | None = None) -> dict:
    """Kernel-smoothed empirical scoreline distributions per outcome and gap bin."""
    fav_goals = np.minimum(fav_goals, MAX_GOALS)
    dog_goals = np.minimum(dog_goals, MAX_GOALS)
    centers = (np.arange(N_BINS) + 0.5) * X_BIN
    raw = np.zeros((N_BINS, MAX_GOALS + 1, MAX_GOALS + 1))
    b = np.minimum((x / X_BIN).astype(int), N_BINS - 1)
    np.add.at(raw, (b, fav_goals, dog_goals), 1.0 if weights is None else weights)
    w = np.exp(-0.5 * ((centers[:, None] - centers[None, :]) / KERNEL_BW) ** 2)
    smooth = np.einsum("kb,bij->kij", w, raw)
    out = {}
    fav = np.greater.outer(np.arange(MAX_GOALS + 1), np.arange(MAX_GOALS + 1))
    for name, mask in (("fav", fav), ("draw", np.eye(MAX_GOALS + 1, dtype=bool)), ("dog", fav.T)):
        t = smooth * mask
        out[name] = t / np.clip(t.sum(axis=(1, 2), keepdims=True), 1e-12, None)
    return out


def fit(d: np.ndarray, var: np.ndarray, g1: np.ndarray, g2: np.ndarray, ot: np.ndarray, playoff: np.ndarray,
        source: np.ndarray, fit_mask: np.ndarray, weights: np.ndarray | None = None) -> GameModel:
    """Fit every part of the game model on the games selected by ``fit_mask``.

    ``d``/``var`` are the point-estimate gap and its posterior variance; likelihoods
    integrate over that posterior so the parameters describe true skill gaps (the
    simulator draws ratings from the posterior). ``weights`` (e.g. recency) apply to all parts.
    """
    w_all = np.ones(len(d)) if weights is None else weights
    lo = np.minimum(g1, g2)
    rg1 = np.where(ot, lo, g1)
    rg2 = np.where(ot, lo, g2)
    y = np.where(rg1 > rg2, 0, np.where(rg1 == rg2, 1, 2))
    rr = fit_mask & ~playoff
    po = fit_mask & playoff
    beta_rr, c_rr = ordered_logit.fit(d[rr], y[rr], weights=w_all[rr], var=var[rr])
    beta_po, c_po = ordered_logit.fit(d[po], y[po], weights=w_all[po], var=var[po])
    c_src = {}
    for s_ in np.unique(source[rr]):
        m = rr & (source == s_)
        c_src[str(s_)] = _fit_c(d[m], y[m], beta_rr, w_all[m] / w_all[m].mean(), var[m])
    ot_games = fit_mask & ot
    beta_ot = _fit_slope(d[ot_games], g1[ot_games] > g2[ot_games], w_all[ot_games], var[ot_games])
    tables = {}
    for kind, m in (("rr", rr), ("po", po)):
        p1_fav = d[m] >= 0
        fav = np.where(p1_fav, rg1[m], rg2[m])
        dog = np.where(p1_fav, rg2[m], rg1[m])
        tables[kind] = scoreline_tables(beta_rr * np.abs(d[m]), fav, dog, w_all[m])
    return GameModel(beta_rr, c_rr, beta_po, c_po, beta_ot, c_src, tables)


def simulate(model: GameModel, d: np.ndarray, playoff: bool, rng: np.random.Generator,
             need_winner: bool | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate one game per gap. Returns (goals p1, goals p2, overtime flag)."""
    d = np.asarray(d, float)
    n = len(d)
    p = model.reg_probs(d, playoff=playoff)
    u = rng.random(n)
    y = (u > p[:, 0]).astype(int) + (u > p[:, 0] + p[:, 1])
    p1_fav = d >= 0
    # outcome from the favourite's view: 0 fav wins, 1 draw, 2 dog wins
    fy = np.where(p1_fav, y, 2 - y)
    b = np.minimum((model.beta_rr * np.abs(d) / X_BIN).astype(int), N_BINS - 1)
    fav = np.empty(n, int)
    dog = np.empty(n, int)
    size = MAX_GOALS + 1
    for k, name in enumerate(OUTCOMES):
        m = fy == k
        if not m.any():
            continue
        flat = model.scorelines["po" if playoff else "rr"][name].reshape(N_BINS, -1)
        cdf = np.cumsum(flat[b[m]], axis=1)
        pick = (rng.random(m.sum())[:, None] > cdf).sum(axis=1)
        pick = np.minimum(pick, size * size - 1)
        fav[m], dog[m] = pick // size, pick % size
    g1 = np.where(p1_fav, fav, dog)
    g2 = np.where(p1_fav, dog, fav)
    need = playoff if need_winner is None else need_winner
    ot = np.zeros(n, bool)
    if need:
        ot = g1 == g2
        p1_wins_ot = rng.random(n) < model.ot_prob(d)
        g1 = g1 + (ot & p1_wins_ot)
        g2 = g2 + (ot & ~p1_wins_ot)
    return g1, g2, ot


def shape_check(model: GameModel, d: np.ndarray, var: np.ndarray, g1: np.ndarray, g2: np.ndarray, ot: np.ndarray,
                playoff: np.ndarray, reps: int = 5, seed: int = 0) -> pd.DataFrame:
    """Real vs simulated draw/OT/goal-difference shares per gap bin.

    Both are binned on the point-estimate gap d; the simulation plays each game ``reps``
    times with a fresh posterior draw of the gap (d + sqrt(var) * z), as the engine does.
    """
    rng = np.random.default_rng(seed)
    bins = [0, 0.25, 0.5, 1, 1.5, 2, 3, np.inf]
    rows = []
    for kind, mask in (("rr", ~playoff), ("po", playoff)):
        dd = d[mask]
        rep_d = np.tile(dd, reps)
        draw = rep_d + np.sqrt(np.tile(var[mask], reps)) * rng.standard_normal(len(rep_d))
        s1, s2, sot = simulate(model, draw, playoff=(kind == "po"), rng=rng)
        # outcomes from the point-estimate favourite's view, like the real games
        real = _shape_stats(dd, g1[mask], g2[mask], ot[mask], kind)
        sim = _shape_stats(rep_d, s1, s2, sot, kind)
        real_bin = pd.cut(model.beta_rr * np.abs(dd), bins, right=False)
        sim_bin = pd.cut(model.beta_rr * np.abs(rep_d), bins, right=False)
        for key in real:
            r = real[key].groupby(real_bin, observed=True).mean()
            s = sim[key].groupby(sim_bin, observed=True).mean()
            for bin_, val in r.items():
                rows.append((kind, key, str(bin_), val, s.get(bin_, np.nan)))
    return pd.DataFrame(rows, columns=["kind", "stat", "x_bin", "real", "sim"])


def _shape_stats(d, g1, g2, ot, kind):
    fav_gd = np.where(d >= 0, g1 - g2, g2 - g1)
    stats = {
        "fav_win": pd.Series(fav_gd > 0, dtype=float),
        "abs_gd>=3": pd.Series(np.abs(g1 - g2) >= 3, dtype=float),
        "total_goals": pd.Series(g1 + g2, dtype=float),
        "fav_gd": pd.Series(fav_gd, dtype=float),
    }
    if kind == "rr":
        stats["draw"] = pd.Series(g1 == g2, dtype=float)
    else:
        stats["ot"] = pd.Series(ot, dtype=float)
    return stats
