"""Walk-forward rating systems.

Every system reads games in chronological order and records, for each game, both
players' rating mean and variance *before* the game, so any prediction made from
them only uses earlier games. Kernels are numba-compiled loops over integer arrays
prepared once by :func:`prepare`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numba as nb
import numpy as np
import pandas as pd

LN10_400 = np.log(10.0) / 400.0


@dataclass
class Prepared:
    a: np.ndarray  # player index of p1
    b: np.ndarray  # player index of p2
    day: np.ndarray  # days since epoch
    gd: np.ndarray  # final goal difference p1 - p2 (OT goal included)
    score: np.ndarray  # 1 / 0.5 / 0 from p1's view
    skip: np.ndarray  # walkovers
    level: np.ndarray  # Scorpion tournament level, -1 if unknown
    birth_day: np.ndarray  # per player, days since epoch, large if unknown
    players: np.ndarray  # player ids by index


def prepare(games: pd.DataFrame, players: pd.DataFrame | None = None) -> Prepared:
    p1 = games["p1"].to_numpy()
    p2 = games["p2"].to_numpy()
    codes, uniques = pd.factorize(np.concatenate([p1, p2]))
    n = len(p1)
    g1 = games["g1"].to_numpy()
    g2 = games["g2"].to_numpy()
    birth = np.full(len(uniques), 10**7, dtype=np.int64)
    if players is not None:
        bd = players.set_index("player_id")["birth_date"]
        bd = pd.to_datetime(pd.Series(uniques).map(bd))
        known = bd.notna().to_numpy()
        birth[known] = (bd[known].to_numpy().astype("datetime64[D]").astype(np.int64))
    level = pd.to_numeric(games.get("level"), errors="coerce") if "level" in games else None
    return Prepared(
        a=codes[:n].astype(np.int64),
        b=codes[n:].astype(np.int64),
        day=games["date"].to_numpy().astype("datetime64[D]").astype(np.int64),
        gd=(g1 - g2).astype(np.float64),
        score=np.where(g1 > g2, 1.0, np.where(g1 < g2, 0.0, 0.5)),
        skip=games["walkover"].to_numpy().astype(np.bool_),
        level=(level.fillna(-1).to_numpy().astype(np.int64) if level is not None else np.full(n, -1, np.int64)),
        birth_day=birth,
        players=np.asarray(uniques),
    )


# --- Elo -------------------------------------------------------------------------


@dataclass
class EloParams:
    k: float = 24.0
    k_new: float = 64.0
    new_games: float = 40.0
    mov: float = 0.0  # 0 = plain Elo; >0 scales the update by goal margin (538-style)


@nb.njit(cache=True)
def _elo(a, b, gd, score, skip, n_players, k, k_new, new_games, mov):
    r = np.full(n_players, 1500.0)
    cnt = np.zeros(n_players, np.int64)
    m1 = np.empty(len(a))
    m2 = np.empty(len(a))
    for i in range(len(a)):
        x, y = a[i], b[i]
        m1[i], m2[i] = r[x], r[y]
        if skip[i]:
            continue
        e = 1.0 / (1.0 + 10.0 ** ((r[y] - r[x]) / 400.0))
        delta = score[i] - e
        if mov > 0.0 and gd[i] != 0.0:
            diff = r[x] - r[y] if gd[i] > 0 else r[y] - r[x]
            delta *= mov * np.log(abs(gd[i]) + 1.0) * 2.2 / (diff * 0.001 + 2.2)
        kx = k + (k_new - k) * max(0.0, 1.0 - cnt[x] / new_games)
        ky = k + (k_new - k) * max(0.0, 1.0 - cnt[y] / new_games)
        r[x] += kx * delta
        r[y] -= ky * delta
        cnt[x] += 1
        cnt[y] += 1
    return m1, m2


# --- Glicko-2 (per-game updates, rating deviation grows with inactivity) ---------


@dataclass
class GlickoParams:
    tau: float = 0.5
    sigma0: float = 0.06
    phi0: float = 350.0
    c_day: float = 2.0  # RD growth per day of inactivity (Glicko-1 style c, rating units)
    phi_min: float = 30.0


@nb.njit(cache=True)
def _vol_f(xx, delta, ph2, v, A, tau):
    ex = np.exp(xx)
    return ex * (delta**2 - ph2 - v - ex) / (2.0 * (ph2 + v + ex) ** 2) - (xx - A) / tau**2


@nb.njit(cache=True)
def _glicko2(a, b, day, score, skip, n_players, tau, sigma0, phi0, c_day, phi_min):
    scale = 173.7178
    mu = np.zeros(n_players)
    phi = np.full(n_players, phi0 / scale)
    sig = np.full(n_players, sigma0)
    last = np.full(n_players, -1, np.int64)
    m1 = np.empty(len(a))
    m2 = np.empty(len(a))
    v1 = np.empty(len(a))
    v2 = np.empty(len(a))
    cday = (c_day / scale) ** 2
    pmax = phi0 / scale
    pmin = phi_min / scale
    for i in range(len(a)):
        for p in (a[i], b[i]):
            if last[p] >= 0 and day[i] > last[p]:
                phi[p] = min(pmax, np.sqrt(phi[p] ** 2 + cday * (day[i] - last[p])))
            last[p] = day[i]
        x, y = a[i], b[i]
        m1[i], m2[i] = mu[x] * scale + 1500.0, mu[y] * scale + 1500.0
        v1[i], v2[i] = (phi[x] * scale) ** 2, (phi[y] * scale) ** 2
        if skip[i]:
            continue
        new = np.empty(3)
        for side in range(2):
            p = x if side == 0 else y
            o = y if side == 0 else x
            s = score[i] if side == 0 else 1.0 - score[i]
            g = 1.0 / np.sqrt(1.0 + 3.0 * phi[o] ** 2 / np.pi**2)
            e = 1.0 / (1.0 + np.exp(-g * (mu[p] - mu[o])))
            v = 1.0 / (g * g * e * (1.0 - e))
            delta = v * g * (s - e)
            # volatility update (Illinois algorithm)
            A = np.log(sig[p] ** 2)
            ph2 = phi[p] ** 2

            lo = A
            if delta**2 > ph2 + v:
                hi = np.log(delta**2 - ph2 - v)
            else:
                k = 1.0
                while _vol_f(A - k * tau, delta, ph2, v, A, tau) < 0:
                    k += 1.0
                hi = A - k * tau
            fa, fb = _vol_f(lo, delta, ph2, v, A, tau), _vol_f(hi, delta, ph2, v, A, tau)
            for _ in range(60):
                if abs(hi - lo) < 1e-6:
                    break
                cc = lo + (lo - hi) * fa / (fb - fa)
                fc = _vol_f(cc, delta, ph2, v, A, tau)
                if fc * fb <= 0:
                    lo, fa = hi, fb
                else:
                    fa = fa / 2.0
                hi, fb = cc, fc
            f_a = np.exp(lo / 2.0)
            phistar = np.sqrt(ph2 + f_a**2)
            phinew = max(pmin, 1.0 / np.sqrt(1.0 / phistar**2 + 1.0 / v))
            munew = mu[p] + phinew**2 * g * (s - e)
            if side == 0:
                new[0], new[1], new[2] = munew, phinew, f_a
            else:
                mu[x], phi[x], sig[x] = new[0], new[1], new[2]
                mu[y], phi[y], sig[y] = munew, phinew, f_a
    return m1, m2, v1, v2


# --- Glicko-2 on a margin-of-victory score (h2h.puck.no style) ------------------
#
# h2h.puck.no (methodology at https://h2h.puck.no/ratings) runs Glicko-2 game by game
# with a soft score s = 1 / (1 + exp(-mov_k * goal difference)) instead of 1 / 0.5 / 0,
# and grows RD before each game by one volatility step per ``week_days`` of inactivity
# (the standard Glicko-2 step adds one more for the game itself).


@dataclass
class GlickoMovParams:
    tau: float = 1.5
    sigma0: float = 0.3
    phi0: float = 350.0
    mov_k: float = 0.4
    week_days: float = 7.0
    gd_cap: float = 10.0
    prior_mean: float = 0.0  # newcomer rating relative to 1500 (puck.no: 0)


@nb.njit(cache=True)
def _glicko2_mov(a, b, day, gd, skip, n_players, tau, sigma0, phi0, mov_k, week_days, gd_cap, prior_mean):
    scale = 173.7178
    mu = np.full(n_players, prior_mean / scale)
    phi = np.full(n_players, phi0 / scale)
    sig = np.full(n_players, sigma0)
    last = np.full(n_players, -1, np.int64)
    m1 = np.empty(len(a))
    m2 = np.empty(len(a))
    v1 = np.empty(len(a))
    v2 = np.empty(len(a))
    pmax = phi0 / scale
    for i in range(len(a)):
        for p in (a[i], b[i]):
            if last[p] >= 0 and day[i] > last[p]:
                weeks = np.floor((day[i] - last[p]) / week_days)
                phi[p] = min(pmax, np.sqrt(phi[p] ** 2 + weeks * sig[p] ** 2))
            last[p] = day[i]
        x, y = a[i], b[i]
        m1[i], m2[i] = mu[x] * scale + 1500.0, mu[y] * scale + 1500.0
        v1[i], v2[i] = (phi[x] * scale) ** 2, (phi[y] * scale) ** 2
        if skip[i]:
            continue
        s1 = 1.0 / (1.0 + np.exp(-mov_k * min(gd_cap, max(-gd_cap, gd[i]))))
        new = np.empty(3)
        for side in range(2):
            p = x if side == 0 else y
            o = y if side == 0 else x
            s = s1 if side == 0 else 1.0 - s1
            g = 1.0 / np.sqrt(1.0 + 3.0 * phi[o] ** 2 / np.pi**2)
            e = 1.0 / (1.0 + np.exp(-g * (mu[p] - mu[o])))
            v = 1.0 / (g * g * e * (1.0 - e))
            delta = v * g * (s - e)
            A = np.log(sig[p] ** 2)
            ph2 = phi[p] ** 2
            lo = A
            if delta**2 > ph2 + v:
                hi = np.log(delta**2 - ph2 - v)
            else:
                k = 1.0
                while _vol_f(A - k * tau, delta, ph2, v, A, tau) < 0:
                    k += 1.0
                hi = A - k * tau
            fa, fb = _vol_f(lo, delta, ph2, v, A, tau), _vol_f(hi, delta, ph2, v, A, tau)
            for _ in range(60):
                if abs(hi - lo) < 1e-6:
                    break
                cc = lo + (lo - hi) * fa / (fb - fa)
                fc = _vol_f(cc, delta, ph2, v, A, tau)
                if fc * fb <= 0:
                    lo, fa = hi, fb
                else:
                    fa = fa / 2.0
                hi, fb = cc, fc
            f_a = np.exp(lo / 2.0)
            phistar = min(pmax, np.sqrt(ph2 + f_a**2))
            phinew = 1.0 / np.sqrt(1.0 / phistar**2 + 1.0 / v)
            munew = mu[p] + phinew**2 * g * (s - e)
            if side == 0:
                new[0], new[1], new[2] = munew, phinew, f_a
            else:
                mu[x], phi[x], sig[x] = new[0], new[1], new[2]
                mu[y], phi[y], sig[y] = munew, phinew, f_a
    return m1, m2, v1, v2, mu, phi, sig, last


# --- Gaussian goal-difference filter (online TrueSkill-through-time) --------------


@dataclass
class KalmanParams:
    obs_sd: float = 2.6  # noise sd of one game's goal difference given the skill gap
    prior_sd: float = 1.2  # sd of a newcomer's skill (goal units)
    prior_mean: float = -0.8  # newcomer mean, relative to the established average 0
    drift_sd_year: float = 0.35  # skill drift sd over a year of inactivity
    drift_floor_day: float = 0.0  # extra variance added per game day (form change)
    junior_age: float = 20.0
    junior_mult: float = 2.0  # drift variance multiplier for players under junior_age
    gd_cap: float = 6.0  # goal differences are clipped to +-cap
    level_prior: float = 0.0  # newcomer mean shift per Scorpion level step below 3


@nb.njit(cache=True)
def _kalman(a, b, day, gd, skip, level, birth_day, n_players, obs_sd, prior_sd, prior_mean,
            drift_sd_year, drift_floor_day, junior_age, junior_mult, gd_cap, level_prior):
    mu = np.zeros(n_players)
    var = np.full(n_players, prior_sd**2)
    last = np.full(n_players, -1, np.int64)
    m1 = np.empty(len(a))
    m2 = np.empty(len(a))
    v1 = np.empty(len(a))
    v2 = np.empty(len(a))
    q_day = drift_sd_year**2 / 365.0
    r = obs_sd**2
    vmax = prior_sd**2
    for i in range(len(a)):
        for p in (a[i], b[i]):
            if last[p] < 0:
                shift = 0.0
                if level[i] > 0:
                    shift = -level_prior * (level[i] - 3)
                mu[p] = prior_mean + shift
                var[p] = prior_sd**2
            elif day[i] > last[p]:
                q = q_day * (day[i] - last[p]) + drift_floor_day
                age = (day[i] - birth_day[p]) / 365.25
                if age > 0 and age < junior_age:
                    q *= junior_mult
                var[p] = min(vmax, var[p] + q)
            last[p] = day[i]
        x, y = a[i], b[i]
        m1[i], m2[i], v1[i], v2[i] = mu[x], mu[y], var[x], var[y]
        if skip[i]:
            continue
        obs = min(gd_cap, max(-gd_cap, gd[i]))
        s = var[x] + var[y] + r
        e = obs - (mu[x] - mu[y])
        mu[x] += var[x] / s * e
        mu[y] -= var[y] / s * e
        var[x] -= var[x] ** 2 / s
        var[y] -= var[y] ** 2 / s
    return m1, m2, v1, v2, mu, var, last


def run(model: str, prep: Prepared, params) -> pd.DataFrame:
    """Run a rating system. Returns pre-game mu1, mu2, var1, var2 (var 0 if none)."""
    n = len(prep.players)
    if model in ("elo", "mov_elo"):
        m1, m2 = _elo(prep.a, prep.b, prep.gd, prep.score, prep.skip, n, *asdict(params).values())
        v1 = v2 = np.zeros(len(m1))
    elif model == "glicko2":
        m1, m2, v1, v2 = _glicko2(prep.a, prep.b, prep.day, prep.score, prep.skip, n, *asdict(params).values())
    elif model == "glicko2_mov":
        m1, m2, v1, v2, *_ = _glicko2_mov(prep.a, prep.b, prep.day, prep.gd, prep.skip, n, *asdict(params).values())
    elif model == "kalman":
        m1, m2, v1, v2, *_ = _kalman(prep.a, prep.b, prep.day, prep.gd, prep.skip, prep.level, prep.birth_day, n,
                                     *asdict(params).values())
    elif model == "tw_bt":
        from thsim.ratings.tw_bt import run_tw_bt

        return run_tw_bt(prep, params)
    else:
        raise ValueError(model)
    return pd.DataFrame({"mu1": m1, "mu2": m2, "var1": v1, "var2": v2})


@nb.njit(cache=True)
def prior_counts(a, b, skip):
    """Played games (walkovers excluded) each player had before every game."""
    n = max(a.max(), b.max()) + 1
    cnt = np.zeros(n, np.int64)
    n1 = np.empty(len(a), np.int64)
    n2 = np.empty(len(a), np.int64)
    for i in range(len(a)):
        n1[i], n2[i] = cnt[a[i]], cnt[b[i]]
        if not skip[i]:
            cnt[a[i]] += 1
            cnt[b[i]] += 1
    return n1, n2


def kalman_final(prep: Prepared, params: KalmanParams, as_of_day: int) -> pd.DataFrame:
    """Every player's current rating: mean, sd (with drift up to ``as_of_day``), games, last game day."""
    *_, mu, var, last = _kalman(prep.a, prep.b, prep.day, prep.gd, prep.skip, prep.level, prep.birth_day,
                                len(prep.players), *asdict(params).values())
    q = params.drift_sd_year**2 / 365.0 * np.maximum(0, as_of_day - last)
    age = (as_of_day - prep.birth_day) / 365.25
    q = np.where((age > 0) & (age < params.junior_age), q * params.junior_mult, q)
    var = np.minimum(params.prior_sd**2, var + q)
    played = np.bincount(np.concatenate([prep.a, prep.b]), minlength=len(prep.players))
    return pd.DataFrame({"player_id": prep.players, "mu": mu, "sd": np.sqrt(var), "games": played, "last_day": last})


def glicko_mov_final(prep: Prepared, params: GlickoMovParams, as_of_day: int) -> pd.DataFrame:
    """Every player's current Glicko-MoV rating: mean and RD (grown to ``as_of_day``), rating points."""
    *_, mu, phi, sig, last = _glicko2_mov(prep.a, prep.b, prep.day, prep.gd, prep.skip, len(prep.players),
                                          *asdict(params).values())
    scale = 173.7178
    weeks = np.floor(np.maximum(0, as_of_day - last) / params.week_days)
    phi = np.minimum(params.phi0 / scale, np.sqrt(phi**2 + weeks * sig**2))
    return pd.DataFrame({"player_id": prep.players, "r": mu * scale + 1500.0, "rd": phi * scale})
