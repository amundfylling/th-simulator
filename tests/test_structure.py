from conftest import make_games
from thsim.structure import assign_groups, group_summary, playoff_series, round_formats, top_places


def test_groups_split_by_connected_components():
    rows = [(1, 2, 1, 0), (1, 3, 1, 1), (2, 3, 0, 2), (4, 5, 2, 2), (4, 6, 1, 0)]
    g = make_games(rows)
    labels = assign_groups(g)
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4] != labels[0]
    s = group_summary(g).sort_values("n_games")
    assert s["complete"].tolist() == [False, True]  # {4,5,6} misses 5-6


def _series(p1, p2, results, round_, **kw):
    return [dict(p1=p1, p2=p2, g1=a, g2=b, round=round_, game_no=i + 1, ot=ot, **kw)
            for i, (a, b, ot) in enumerate(results)]


def test_series_length_inference_and_top_places():
    rows = []
    rows += _series(1, 4, [(3, 1, False), (2, 3, True), (4, 0, False)], 0.5)  # 1 wins 2-1
    rows += _series(2, 3, [(0, 2, False), (1, 2, False)], 0.5)  # 3 wins 2-0
    rows += _series(4, 2, [(2, 1, False), (2, 2, False), (3, 2, True)], 0.9)  # draw allowed, 4 wins 2-0
    rows += _series(1, 3, [(1, 0, False), (0, 1, False), (5, 4, True)], 1.0)  # 1 wins 2-1
    g = make_games(rows, stage_kind="po", stage_uid=20)
    s = playoff_series(g)
    assert len(s) == 4
    bronze = s.loc[s["round"].eq(0.9)].iloc[0]
    assert (bronze["winner"], bronze["winner_wins"], bronze["draws"]) == (4, 2, 1)
    f = round_formats(s)
    assert f["wins_needed"].eq(2).all() and f["consistent"].all()
    places = top_places(s)
    assert places.set_index("player")["place"].to_dict() == {1: 1, 3: 2, 4: 3, 2: 4}
