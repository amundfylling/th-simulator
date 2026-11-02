import numpy as np
import pandas as pd

from conftest import make_games
from thsim.ratings import models as M


def _games(n=3000, seed=0):
    """Round-robin games among 6 players whose true skills differ by one goal each."""
    rng = np.random.default_rng(seed)
    skill = np.arange(6, dtype=float)
    rows = []
    for k in range(n):
        a, b = rng.choice(6, 2, replace=False)
        gd = int(np.round(skill[a] - skill[b] + rng.normal(0, 1.5)))
        rows.append(dict(p1=int(a) + 1, p2=int(b) + 1, g1=max(gd, 0) + 1, g2=max(-gd, 0) + 1,
                         date=pd.Timestamp("2020-01-01") + pd.Timedelta(days=k // 20)))
    return make_games(rows)


def test_kalman_recovers_skill_order_and_shrinks_uncertainty():
    games = _games()
    prep = M.prepare(games)
    final = M.kalman_final(prep, M.KalmanParams(), as_of_day=int(prep.day.max()))
    final = final.sort_values("player_id")
    assert list(np.argsort(final["mu"].to_numpy())) == list(range(6))
    gaps = np.diff(final["mu"].to_numpy())
    assert np.all(np.abs(gaps - 1.0) < 0.35)
    assert final["sd"].max() < 0.3


def test_ratings_are_walk_forward():
    games = _games(200)
    prep = M.prepare(games)
    full = M.run("kalman", prep, M.KalmanParams())
    head = M.run("kalman", M.prepare(games.iloc[:100]), M.KalmanParams())
    # Pre-game ratings of the first 100 games do not depend on later games.
    np.testing.assert_allclose(full.iloc[:100][["mu1", "mu2"]].to_numpy(), head[["mu1", "mu2"]].to_numpy())


def test_uncertainty_grows_with_inactivity():
    rows = [dict(p1=1, p2=2, g1=3, g2=1, date=pd.Timestamp("2020-01-01") + pd.Timedelta(days=k)) for k in range(50)]
    rows.append(dict(p1=1, p2=2, g1=3, g2=1, date=pd.Timestamp("2023-01-01")))
    r = M.run("kalman", M.prepare(make_games(rows)), M.KalmanParams())
    assert r["var1"].iloc[-1] > 3 * r["var1"].iloc[-2]


def test_elo_and_glicko_favour_the_winner():
    games = make_games([(1, 2, 3, 0)] * 20)
    prep = M.prepare(games)
    for model, params in [("elo", M.EloParams()), ("mov_elo", M.EloParams(mov=1.0)), ("glicko2", M.GlickoParams())]:
        r = M.run(model, prep, params)
        assert r["mu1"].iloc[-1] > r["mu2"].iloc[-1]


def test_prior_counts_skip_walkovers():
    games = make_games([dict(p1=1, p2=2, g1=5, g2=0, walkover=True), (1, 2, 3, 1), (1, 3, 2, 2)])
    prep = M.prepare(games)
    n1, n2 = M.prior_counts(prep.a, prep.b, prep.skip)
    # The walkover is not history: both players are new at their first played game.
    assert (n1[1], n2[1]) == (0, 0)
    assert n1[2] == 1
