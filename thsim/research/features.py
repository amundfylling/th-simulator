"""Pre-game features for every game, walk-forward (M9).

Every feature uses only games played before the one it describes. Features come per side
(``_1`` for p1, ``_2`` for p2) so a model can be made order-free by mirroring: the same
game seen from p2's side swaps the two columns and flips the outcome.

With ``freeze=True`` the per-event features (head-to-head edge, form, games played) stand as
they were before the event's first game, which is what a pre-tournament forecast sees.
"""

from __future__ import annotations

import numba as nb
import numpy as np
import pandas as pd

from thsim.backtest_games import Context
from thsim.ratings import composite as C
from thsim.ratings import models as M

SCALE = C.SCALE


@nb.njit(cache=True)
def _player_running(a, b, day, skip, rr, draw, resid, event, n_players, hl_games, freeze):
    """Per game and side: games played, rr games, rr draws, EWMA residual (form), last game day.

    ``freeze``: values stay as they were at each player's first game in the current event.
    """
    games = np.zeros(n_players)
    rr_n = np.zeros(n_players)
    rr_d = np.zeros(n_players)
    form = np.zeros(n_players)
    last = np.full(n_players, -1, np.int64)
    snap_ev = np.full(n_players, -1, np.int64)
    snap = np.zeros((n_players, 5))
    out = np.zeros((len(a), 2, 5))
    decay = 0.5 ** (1.0 / hl_games)
    for i in range(len(a)):
        for s in range(2):
            p = a[i] if s == 0 else b[i]
            cur = np.array([games[p], rr_n[p], rr_d[p], form[p], day[i] - last[p] if last[p] >= 0 else -1.0])
            if freeze:
                if snap_ev[p] != event[i]:
                    snap_ev[p] = event[i]
                    snap[p] = cur
                out[i, s] = snap[p]
            else:
                out[i, s] = cur
        if skip[i]:
            continue
        for s in range(2):
            p = a[i] if s == 0 else b[i]
            r = resid[i] if s == 0 else -resid[i]
            games[p] += 1
            if rr[i]:
                rr_n[p] += 1
                rr_d[p] += draw[i]
            form[p] = decay * form[p] + (1 - decay) * r
            last[p] = day[i]
    return out


@nb.njit(cache=True)
def _event_form(a, b, skip, resid, event, n_players):
    """Per game and side: games and summed residual earlier in the same event."""
    ev = np.full(n_players, -1, np.int64)
    n = np.zeros(n_players)
    s = np.zeros(n_players)
    out = np.zeros((len(a), 2, 2))
    for i in range(len(a)):
        for k in range(2):
            p = a[i] if k == 0 else b[i]
            if ev[p] != event[i]:
                ev[p] = event[i]
                n[p] = 0.0
                s[p] = 0.0
            out[i, k, 0] = n[p]
            out[i, k, 1] = s[p]
        if skip[i]:
            continue
        n[a[i]] += 1
        n[b[i]] += 1
        s[a[i]] += resid[i]
        s[b[i]] -= resid[i]
    return out


def build(ctx: Context, games: pd.DataFrame, kp: M.KalmanParams, cp: C.CompositeParams, freeze: bool = False,
          form_hl: float = 30.0) -> pd.DataFrame:
    prep = ctx.prep
    event = games["tournament_uid"].factorize()[0].astype(np.int64)
    k = M.run("kalman", prep, kp)
    g = M.run("glicko2_mov", prep, cp.glicko)
    r = C.run(prep, kp, cp)
    elo = M.run("elo", prep, M.EloParams())
    d = (r["mu1"] - r["mu2"]).to_numpy()
    resid_score = C.residuals(prep, r, cp)
    edge, meet = C.h2h_edge(prep, resid_score, event, cp.h2h_prior, freeze=freeze)
    # Only the pairs the website gets (2+ meetings, |edge| >= 0.01 goals) carry head-to-head features.
    edge = cp.h2h_coef * edge
    kept = (meet >= C.H2H_MIN_MEETINGS) & (np.abs(edge) >= C.H2H_MIN_ABS)
    edge = np.where(kept, edge, 0.0)
    meet = np.where(kept, meet, 0.0)
    gd_resid = np.where(prep.skip, 0.0, np.clip(prep.gd, -6, 6) - d)
    rr = games["stage_kind"].eq("rr").to_numpy()
    draw = (games["g1"] == games["g2"]).to_numpy().astype(float)
    run = _player_running(prep.a, prep.b, prep.day, prep.skip, rr, draw, gd_resid, event, len(prep.players), form_hl, freeze)
    evf = _event_form(prep.a, prep.b, prep.skip, gd_resid, event, len(prep.players))
    if freeze:
        evf[:] = 0.0
    age = lambda bd: np.where(bd < 10**6, (prep.day - bd) / 365.25, np.nan)  # noqa: E731
    f = pd.DataFrame({
        "d": d,
        "sd": np.sqrt((r["var1"] + r["var2"]).to_numpy()),
        "h2h": edge,
        "meet": meet,
        "po": games["stage_kind"].eq("po").to_numpy().astype(float),
        "level": prep.level.astype(float),
        "bordshockey": (games["source"] == "bordshockey").to_numpy().astype(float),
        "team": games["team_event"].to_numpy().astype(float),
    })
    for s, (mu, var) in enumerate((("mu1", "var1"), ("mu2", "var2")), start=1):
        side = s - 1
        f[f"mu_{s}"] = r[mu].to_numpy()
        f[f"kal_{s}"] = k[mu].to_numpy()
        f[f"kal_sd_{s}"] = np.sqrt(k[var].to_numpy())
        f[f"gl_{s}"] = (g[mu].to_numpy() - 1500) / SCALE
        f[f"rd_{s}"] = np.sqrt(g[var].to_numpy()) / SCALE
        f[f"elo_{s}"] = (elo[mu].to_numpy() - 1500) / 100
        f[f"games_{s}"] = np.log1p(run[:, side, 0])
        f[f"draw_rate_{s}"] = (run[:, side, 2] + 0.125 * 20) / (run[:, side, 1] + 20)
        f[f"form_{s}"] = run[:, side, 3]
        f[f"days_{s}"] = np.log1p(np.maximum(run[:, side, 4], 0)) * (run[:, side, 4] >= 0)
        f[f"ev_games_{s}"] = evf[:, side, 0]
        f[f"ev_form_{s}"] = evf[:, side, 1] / (evf[:, side, 0] + 4)
        f[f"age_{s}"] = age(prep.birth_day[prep.a if s == 1 else prep.b])
    return f


SIDE_COLS = ["mu", "kal", "kal_sd", "gl", "rd", "elo", "games", "draw_rate", "form", "days", "ev_games", "ev_form", "age"]
ANTI_COLS = ["d", "h2h"]  # flip sign when the sides swap


def mirror(f: pd.DataFrame) -> pd.DataFrame:
    """The same games seen from p2's side."""
    m = f.copy()
    for c in SIDE_COLS:
        m[f"{c}_1"], m[f"{c}_2"] = f[f"{c}_2"].to_numpy(), f[f"{c}_1"].to_numpy()
    for c in ANTI_COLS:
        m[c] = -f[c]
    return m
