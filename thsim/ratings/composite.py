"""Composite rating: our Kalman goal-difference filter blended with an h2h.puck.no-style
Glicko-2 on a margin score, plus a head-to-head edge per pair.

* mu = (1 - w) * mu_kalman + w * scale * (r - 1500) / 173.7      (goal units)
* sd = sd_mult * ((1 - w) * sd_kalman + w * scale * RD / 173.7)   (the two are highly correlated;
  sd_mult < 1 because Glicko's RD overstates the posterior spread, fitted on 2015-2021)
* pair edge (goal units) = coef * sum(score - expected) / (meetings + prior), from earlier games
  between the two players; ``score`` is 1 / 0.5 / 0 and ``expected`` comes from the regulation
  model on the composite gap (the h2h.puck.no definition, https://h2h.puck.no/ratings).

Parameters are fitted by ``python -m thsim.research.puck fit`` and stored in reports/m8_params.json.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numba as nb
import numpy as np
import pandas as pd
from numba import types
from numba.typed import Dict

from thsim.ratings import models as M

SCALE = 173.7178
PARAMS_PATH = Path("reports/m8_params.json")


@dataclass
class CompositeParams:
    glicko: M.GlickoMovParams = field(default_factory=M.GlickoMovParams)
    weight: float = 0.0  # share of the Glicko-MoV rating in the blend
    sd_mult: float = 1.0  # scales the blended sd
    scale: float = 1.0  # goals per Glicko unit (173.7 rating points)
    h2h_coef: float = 0.0  # goal units per unit of edge
    h2h_prior: float = 20.0
    # Regulation model used for the expected score in the edge (round-robin beta and draw cut).
    exp_beta: float = 0.8
    exp_c: float = 0.38

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> CompositeParams:
        d = dict(d)
        d["glicko"] = M.GlickoMovParams(**d["glicko"])
        return cls(**d)


def load(path: Path = PARAMS_PATH) -> CompositeParams | None:
    return CompositeParams.from_json(json.loads(path.read_text())["composite"]) if path.exists() else None


def blend(mu_k, var_k, r, rd2, cp: CompositeParams) -> tuple[np.ndarray, np.ndarray]:
    w = cp.weight
    mu = (1 - w) * mu_k + w * cp.scale * (r - 1500.0) / SCALE
    sd = cp.sd_mult * ((1 - w) * np.sqrt(var_k) + w * cp.scale * np.sqrt(rd2) / SCALE)
    return mu, sd**2


def run(prep: M.Prepared, kp: M.KalmanParams, cp: CompositeParams) -> pd.DataFrame:
    """Pre-game composite mu1, mu2, var1, var2 (goal units) for every game."""
    k = M.run("kalman", prep, kp)
    g = M.run("glicko2_mov", prep, cp.glicko)
    mu1, var1 = blend(k["mu1"].to_numpy(), k["var1"].to_numpy(), g["mu1"].to_numpy(), g["var1"].to_numpy(), cp)
    mu2, var2 = blend(k["mu2"].to_numpy(), k["var2"].to_numpy(), g["mu2"].to_numpy(), g["var2"].to_numpy(), cp)
    return pd.DataFrame({"mu1": mu1, "mu2": mu2, "var1": var1, "var2": var2})


def final(prep: M.Prepared, kp: M.KalmanParams, cp: CompositeParams, as_of_day: int) -> pd.DataFrame:
    """Current composite rating per player (same columns as ``M.kalman_final``)."""
    k = M.kalman_final(prep, kp, as_of_day)
    g = M.glicko_mov_final(prep, cp.glicko, as_of_day)
    mu, var = blend(k["mu"].to_numpy(), k["sd"].to_numpy() ** 2, g["r"].to_numpy(), g["rd"].to_numpy() ** 2, cp)
    return k.assign(mu=mu, sd=np.sqrt(var))


def newcomer(kp: M.KalmanParams, cp: CompositeParams) -> dict:
    """Newcomer prior on the composite scale (the website uses it for unrated entrants)."""
    mu = (1 - cp.weight) * kp.prior_mean + cp.weight * cp.scale * cp.glicko.prior_mean / SCALE
    sd = cp.sd_mult * ((1 - cp.weight) * kp.prior_sd + cp.weight * cp.scale * cp.glicko.phi0 / SCALE)
    return {"prior_mean": mu, "prior_sd": sd, "level_prior": (1 - cp.weight) * kp.level_prior}


# --- head-to-head edge ----------------------------------------------------------------


def expected_score(d: np.ndarray, var: np.ndarray, beta: float, c: float) -> np.ndarray:
    """p1's expected score (win 1, draw 0.5) from the regulation model on the gap."""
    from thsim.models import ordered_logit

    p = ordered_logit.integrated_probs(d, var, beta, c)
    return p[:, 0] + 0.5 * p[:, 1]


@nb.njit(cache=True)
def _h2h(a, b, resid, skip, event, n_players, prior, freeze):
    """Per game: the pair's earlier summed residual / (meetings + prior), from p1's view.

    ``resid`` is p1's result minus expectation. With ``freeze`` the pair's meetings in the
    current event are left out (a pre-tournament forecast must not see them).
    """
    tot = Dict.empty(key_type=types.int64, value_type=types.float64)
    cnt = Dict.empty(key_type=types.int64, value_type=types.float64)
    ev_tot = Dict.empty(key_type=types.int64, value_type=types.float64)
    ev_cnt = Dict.empty(key_type=types.int64, value_type=types.float64)
    ev_id = Dict.empty(key_type=types.int64, value_type=types.int64)
    out = np.zeros(len(a))
    meet = np.zeros(len(a))
    for i in range(len(a)):
        lo, hi = (a[i], b[i]) if a[i] < b[i] else (b[i], a[i])
        sign = 1.0 if a[i] == lo else -1.0
        key = lo * n_players + hi
        s = tot.get(key, 0.0)
        n = cnt.get(key, 0.0)
        if freeze and ev_id.get(key, -1) == event[i]:
            s -= ev_tot.get(key, 0.0)
            n -= ev_cnt.get(key, 0.0)
        out[i] = sign * s / (n + prior)
        meet[i] = n
        if skip[i]:
            continue
        tot[key] = tot.get(key, 0.0) + sign * resid[i]
        cnt[key] = cnt.get(key, 0.0) + 1.0
        if ev_id.get(key, -1) != event[i]:
            ev_id[key] = event[i]
            ev_tot[key] = 0.0
            ev_cnt[key] = 0.0
        ev_tot[key] += sign * resid[i]
        ev_cnt[key] += 1.0
    return out, meet


def h2h_edge(prep: M.Prepared, resid: np.ndarray, event: np.ndarray, prior: float, freeze: bool = False):
    """Pre-game pair edge (unscaled) and earlier meetings for every game."""
    resid = np.where(prep.skip, 0.0, resid)
    return _h2h(prep.a, prep.b, resid, prep.skip, event.astype(np.int64), len(prep.players), prior, freeze)


def h2h_final(prep: M.Prepared, resid: np.ndarray, cp: CompositeParams, players: set | None = None,
              min_meetings: int = 2, min_abs: float = 0.01) -> dict[str, float]:
    """Current pair edges in goal units, "idA-idB" (idA < idB) -> shift in idA's favour."""
    ok = ~prep.skip
    pa, pb = prep.players[prep.a[ok]], prep.players[prep.b[ok]]
    r = resid[ok]
    lo = np.minimum(pa, pb)
    hi = np.maximum(pa, pb)
    sign = np.where(pa == lo, 1.0, -1.0)
    df = pd.DataFrame({"lo": lo, "hi": hi, "r": sign * r})
    if players is not None:
        keep = df["lo"].isin(players) & df["hi"].isin(players)
        df = df.loc[keep]
    agg = df.groupby(["lo", "hi"])["r"].agg(["sum", "count"])
    agg = agg.loc[agg["count"] >= min_meetings]
    edge = cp.h2h_coef * agg["sum"] / (agg["count"] + cp.h2h_prior)
    edge = edge.loc[edge.abs() >= min_abs]
    return {f"{int(a)}-{int(b)}": round(float(v), 3) for (a, b), v in edge.items()}


def residuals(prep: M.Prepared, r: pd.DataFrame, cp: CompositeParams) -> np.ndarray:
    d = (r["mu1"] - r["mu2"]).to_numpy()
    var = (r["var1"] + r["var2"]).to_numpy()
    return prep.score - expected_score(d, var, cp.exp_beta, cp.exp_c)

