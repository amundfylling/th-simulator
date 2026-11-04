import numpy as np
import pandas as pd

from conftest import make_games
from thsim.backtest_games import Context
from thsim.models import gbm as G
from thsim.ratings import composite as C
from thsim.ratings import models as M
from thsim.research import features as FT


def _games(n_events=30, seed=0):
    rng = np.random.default_rng(seed)
    skill = np.linspace(-2, 2, 8)
    rows = []
    for e in range(n_events):
        for k in range(12):
            a, b = rng.choice(8, 2, replace=False)
            gd = int(np.round(skill[a] - skill[b] + rng.normal(0, 1.5)))
            rows.append(dict(p1=int(a) + 1, p2=int(b) + 1, g1=max(gd, 0) + 1, g2=max(-gd, 0) + 1, tournament_uid=e + 1,
                             stage_uid=e + 1, date=pd.Timestamp("2019-01-01") + pd.Timedelta(days=7 * e)))
    return make_games(rows)


def _cp():
    return C.CompositeParams(weight=0.7, scale=1.5, sd_mult=0.65, h2h_coef=2.5)


def test_frozen_features_do_not_move_inside_an_event():
    games = _games()
    f = FT.build(Context(games, None), games, M.KalmanParams(), _cp(), freeze=True)
    f["t"] = games["tournament_uid"].to_numpy()
    f["p"] = games["p1"].to_numpy()
    for col in ("games_1", "form_1", "draw_rate_1", "days_1"):
        assert f.groupby(["t", "p"])[col].nunique().max() == 1, col
    assert (f["ev_form_1"] == 0).all()


def test_mirror_is_an_involution():
    games = _games(5)
    f = FT.build(Context(games, None), games, M.KalmanParams(), _cp())
    back = FT.mirror(FT.mirror(f))
    pd.testing.assert_frame_equal(back, f)
    m = FT.mirror(f)
    np.testing.assert_allclose(m["d"], -f["d"])
    np.testing.assert_allclose(m["games_1"], f["games_2"])


def _eval_tree(t, x):
    k = t["root"]
    while k >= 0:
        v = x[t["f"][k]]
        if t["m"][k] == 2 and np.isnan(v):
            left = t["dl"][k] == 1
        else:
            left = (0.0 if np.isnan(v) else v) <= t["t"][k]
        k = t["l"][k] if left else t["r"][k]
    return t["v"][~k]


def test_exported_trees_reproduce_lightgbm_and_probe_features_match_history():
    games = _games()
    ctx = Context(games, None)
    cp = _cp()
    f = FT.build(ctx, games, M.KalmanParams(), cp, freeze=True)
    old = dict(G.PARAMS)
    G.PARAMS.update(min_data_in_leaf=5, num_leaves=7)
    try:
        model = G.train(f, ctx.y3, ctx.known, rounds=5, log=lambda *_: None)
    finally:
        G.PARAMS.clear()
        G.PARAMS.update(old)
    js = G.to_json(model)
    X = f[G.FEATURES].to_numpy(float)[:50]
    raw = np.zeros((50, 3))
    for i, t in enumerate(js["trees"]):
        raw[:, i % 3] += [_eval_tree(t, x) for x in X]
    p = np.exp(raw - raw.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    np.testing.assert_allclose(p, model.predict(f[G.FEATURES].iloc[:50]), atol=1e-5)
    # Probe rows: each player's current games played equals their real count.
    side, newcomer = G.current_features(games, None, M.KalmanParams(), cp, pd.Timestamp("2020-06-01"), [1, 2, 3])
    played = pd.concat([games["p1"], games["p2"]]).value_counts()
    np.testing.assert_allclose(np.expm1(side["games"].to_numpy()), played.loc[[1, 2, 3]].to_numpy())
    assert newcomer["games"] == 0
