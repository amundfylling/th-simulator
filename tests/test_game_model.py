import numpy as np

from thsim.models import game_model


def _synthetic(n=60_000, seed=1):
    rng = np.random.default_rng(seed)
    d = rng.normal(0, 1.5, n)
    true = game_model.GameModel(0.8, 0.37, 0.8, 0.37, 0.4, {}, {})
    # scorelines: favourite wins by 1-3, draws 1-1 or 2-2
    p = true.reg_probs(d)
    u = rng.random(n)
    y = (u > p[:, 0]).astype(int) + (u > p[:, 0] + p[:, 1])
    margin = rng.integers(1, 4, n)
    base = rng.integers(0, 3, n)
    g1 = np.where(y == 0, base + margin, np.where(y == 1, base, base))
    g2 = np.where(y == 2, base + margin, base)
    playoff = rng.random(n) < 0.3
    ot = playoff & (g1 == g2)
    win_ot = rng.random(n) < 1 / (1 + np.exp(-0.4 * d))
    g1 = g1 + (ot & win_ot)
    g2 = g2 + (ot & ~win_ot)
    return d, g1, g2, ot, playoff


def test_fit_recovers_outcome_and_overtime_parameters():
    d, g1, g2, ot, playoff = _synthetic()
    m = game_model.fit(d, np.zeros_like(d), g1, g2, ot, playoff, np.array(["s"] * len(d)), np.ones(len(d), bool))
    assert abs(m.beta_rr - 0.8) < 0.05 and abs(m.c_rr - 0.37) < 0.05
    assert abs(m.beta_ot - 0.4) < 0.1


def test_simulated_shapes_match_the_data_they_came_from():
    d, g1, g2, ot, playoff = _synthetic()
    m = game_model.fit(d, np.zeros_like(d), g1, g2, ot, playoff, np.array(["s"] * len(d)), np.ones(len(d), bool))
    shape = game_model.shape_check(m, d, np.zeros_like(d), g1, g2, ot, playoff, reps=2)
    big = shape.loc[shape["stat"].isin(["fav_win", "draw", "ot"])]
    assert (big["sim"] - big["real"]).abs().max() < 0.03
    # playoff simulations never end level
    s1, s2, _ = game_model.simulate(m, d[:1000], playoff=True, rng=np.random.default_rng(0))
    assert np.all(s1 != s2)
