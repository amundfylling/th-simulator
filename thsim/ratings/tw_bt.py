"""Time-weighted Bradley-Terry style batch ratings, refit at the start of every month.

Each refit solves a weighted ridge regression of (clipped) goal difference on the
skill gap over the last ``window_years`` of games, with weights halving every
``half_life_days``. Games in a month are predicted from the fit made before the month
starts, so this is walk-forward, but coarser than per-tournament refits.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr

from thsim.ratings.models import Prepared


@dataclass
class TwBtParams:
    half_life_days: float = 365.0
    window_years: float = 4.0
    ridge: float = 2.0  # prior precision, in "games" of weight
    gd_cap: float = 6.0
    obs_sd: float = 2.4
    prior_mean: float = -0.8
    start: str = "2014-01-01"


def run_tw_bt(prep: Prepared, params: TwBtParams = TwBtParams()) -> pd.DataFrame:
    n_games = len(prep.a)
    n_players = len(prep.players)
    mu1 = np.full(n_games, params.prior_mean)
    mu2 = np.full(n_games, params.prior_mean)
    v1 = np.full(n_games, params.obs_sd**2 / params.ridge)
    v2 = v1.copy()
    days = prep.day
    months = days.astype("datetime64[D]").astype("datetime64[M]")
    start = np.datetime64(params.start, "M")
    y = np.clip(prep.gd, -params.gd_cap, params.gd_cap)
    ok = ~prep.skip
    for m in np.unique(months[months >= start]):
        cutoff = m.astype("datetime64[D]").astype(np.int64)
        lo = cutoff - int(params.window_years * 365.25)
        train = ok & (days < cutoff) & (days >= lo)
        idx = np.flatnonzero(train)
        target = np.flatnonzero(months == m)
        if len(idx) == 0:
            continue
        w = np.sqrt(0.5 ** ((cutoff - days[idx]) / params.half_life_days))
        rows = np.repeat(np.arange(len(idx)), 2)
        cols = np.column_stack([prep.a[idx], prep.b[idx]]).ravel()
        vals = np.column_stack([w, -w]).ravel()
        X = csr_matrix((vals, (rows, cols)), shape=(len(idx), n_players))
        # Ridge toward the newcomer mean: solve for s - prior_mean. No x0: lsqr would shrink toward it instead.
        sol = lsqr(X, w * y[idx], damp=np.sqrt(params.ridge), atol=1e-6, btol=1e-6, iter_lim=300)[0]
        s = sol + params.prior_mean
        precision = np.bincount(np.concatenate([prep.a[idx], prep.b[idx]]), weights=np.concatenate([w**2, w**2]), minlength=n_players) + params.ridge
        var = params.obs_sd**2 / precision
        mu1[target] = s[prep.a[target]]
        mu2[target] = s[prep.b[target]]
        v1[target] = var[prep.a[target]]
        v2[target] = var[prep.b[target]]
    return pd.DataFrame({"mu1": mu1, "mu2": mu2, "var1": v1, "var2": v2})
