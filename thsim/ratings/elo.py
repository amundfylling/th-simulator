"""Plain sequential Elo, the baseline every other rating model must beat."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class EloParams:
    k: float = 24.0
    k_new: float = 64.0  # K for a player's first game, decaying linearly to k
    new_games: int = 40
    initial: float = 1500.0


def run_elo(games: pd.DataFrame, params: EloParams = EloParams()) -> pd.DataFrame:
    """Walk-forward Elo over chronologically sorted games.

    Returns, aligned with ``games``, each player's rating and number of prior games
    *before* the game. Every game counts (team events included); walkovers are skipped.
    """
    p1 = games["p1"].to_numpy()
    p2 = games["p2"].to_numpy()
    g1 = games["g1"].to_numpy()
    g2 = games["g2"].to_numpy()
    skip = games["walkover"].to_numpy()
    codes, uniques = pd.factorize(np.concatenate([p1, p2]))
    a_idx, b_idx = codes[: len(p1)], codes[len(p1) :]
    rating = np.full(len(uniques), params.initial)
    played = np.zeros(len(uniques), dtype=np.int64)
    r1 = np.empty(len(p1))
    r2 = np.empty(len(p1))
    n1 = np.empty(len(p1), dtype=np.int64)
    n2 = np.empty(len(p1), dtype=np.int64)
    score = np.where(g1 > g2, 1.0, np.where(g1 < g2, 0.0, 0.5))
    k, k_new, new_games = params.k, params.k_new, params.new_games
    rating_l = rating.tolist()
    played_l = played.tolist()
    for i, (a, b) in enumerate(zip(a_idx.tolist(), b_idx.tolist())):
        ra, rb = rating_l[a], rating_l[b]
        na, nb = played_l[a], played_l[b]
        r1[i], r2[i], n1[i], n2[i] = ra, rb, na, nb
        if skip[i]:
            continue
        expected = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
        delta = score[i] - expected
        ka = k + (k_new - k) * max(0.0, 1.0 - na / new_games)
        kb = k + (k_new - k) * max(0.0, 1.0 - nb / new_games)
        rating_l[a] = ra + ka * delta
        rating_l[b] = rb - kb * delta
        played_l[a] = na + 1
        played_l[b] = nb + 1
    return pd.DataFrame({"r1": r1, "r2": r2, "n1": n1, "n2": n2}, index=games.index)
