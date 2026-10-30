"""Infer stage structure from played games: groups, playoff series and their lengths."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def structural(games: pd.DataFrame) -> pd.DataFrame:
    """Games that describe a tournament's structure (drops cross-source leftovers)."""
    if "dedupe_leftover" in games:
        return games.loc[~games["dedupe_leftover"]]
    return games


def assign_groups(games: pd.DataFrame) -> pd.Series:
    """Label every round-robin game with a group id.

    One stage can hold several groups, so a group is a connected component of the
    who-played-whom graph inside the stage. Returns an int64 Series aligned with
    ``games`` (-1 for playoff games).
    """
    out = pd.Series(-1, index=games.index, dtype="int64")
    rr = games.loc[games["stage_kind"].eq("rr"), ["stage_uid", "p1", "p2"]]
    if rr.empty:
        return out
    nodes1 = pd.MultiIndex.from_arrays([rr["stage_uid"], rr["p1"]])
    nodes2 = pd.MultiIndex.from_arrays([rr["stage_uid"], rr["p2"]])
    codes, uniques = pd.factorize(nodes1.append(nodes2))
    a, b = codes[: len(rr)], codes[len(rr) :]
    n = len(uniques)
    graph = coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    out.loc[rr.index] = labels[a]
    return out


def group_summary(games: pd.DataFrame) -> pd.DataFrame:
    """One row per round-robin group: size, games, completeness, legs."""
    rr = structural(games)
    rr = rr.loc[rr["stage_kind"].eq("rr")].copy()
    rr["group"] = assign_groups(rr)
    lo = np.minimum(rr["p1"], rr["p2"])
    hi = np.maximum(rr["p1"], rr["p2"])
    rr["pair"] = lo * 10_000_000 + hi
    agg = rr.groupby("group").agg(
        source=("source", "first"),
        tournament_uid=("tournament_uid", "first"),
        stage_uid=("stage_uid", "first"),
        stage_seq=("stage_seq", "first"),
        date=("date", "min"),
        n_games=("pair", "size"),
        n_pairs=("pair", "nunique"),
        team_event=("team_event", "first"),
    )
    players = pd.concat([rr[["group", "p1"]].rename(columns={"p1": "p"}), rr[["group", "p2"]].rename(columns={"p2": "p"})])
    agg["n_players"] = players.groupby("group")["p"].nunique()
    agg["possible_pairs"] = agg["n_players"] * (agg["n_players"] - 1) // 2
    agg["complete"] = agg["n_pairs"].eq(agg["possible_pairs"])
    agg["legs"] = (agg["n_games"] / agg["n_pairs"]).round(2)
    return agg.reset_index()


def playoff_series(games: pd.DataFrame) -> pd.DataFrame:
    """Collapse playoff games into series (one row per pair per round).

    Series length is not stored, so ``wins_needed`` is inferred as the winner's
    win count (a series is first to N wins). Games drawn without overtime
    (drawn-games-allowed series) count as draws, not wins.
    """
    po = structural(games)
    po = po.loc[po["stage_kind"].eq("po")].copy()
    lo = np.minimum(po["p1"], po["p2"])
    hi = np.maximum(po["p1"], po["p2"])
    lo_goals = np.where(po["p1"].eq(lo), po["g1"], po["g2"])
    hi_goals = np.where(po["p1"].eq(lo), po["g2"], po["g1"])
    po["a"], po["b"] = lo, hi
    po["a_win"] = lo_goals > hi_goals
    po["b_win"] = hi_goals > lo_goals
    po["draw"] = lo_goals == hi_goals
    po["round_key"] = po["round"].fillna(-1.0)
    keys = ["source", "tournament_uid", "stage_uid", "bracket", "stage_name", "round_key", "a", "b"]
    s = po.groupby(keys, sort=False).agg(
        stage_seq=("stage_seq", "first"),
        date=("date", "min"),
        games=("a_win", "size"),
        a_wins=("a_win", "sum"),
        b_wins=("b_win", "sum"),
        draws=("draw", "sum"),
        ot_games=("ot", "sum"),
        walkover=("walkover", "any"),
        first_p1=("p1", "first"),
    )
    s = s.reset_index()
    s["round"] = s["round_key"].replace(-1.0, np.nan)
    a_won = s["a_wins"] > s["b_wins"]
    s["winner"] = np.where(a_won, s["a"], s["b"])
    s["loser"] = np.where(a_won, s["b"], s["a"])
    s["winner_wins"] = np.maximum(s["a_wins"], s["b_wins"])
    s["loser_wins"] = np.minimum(s["a_wins"], s["b_wins"])
    s["decided"] = s["a_wins"] != s["b_wins"]
    return s.drop(columns=["round_key"])


def round_formats(series: pd.DataFrame) -> pd.DataFrame:
    """Wins needed per playoff round: the most wins any series winner reached."""
    ok = series.loc[series["decided"] & ~series["walkover"]].copy()
    ok["round_key"] = ok["round"].fillna(-1.0)
    keys = ["source", "tournament_uid", "stage_uid", "bracket", "stage_name", "round_key"]
    fmt = ok.groupby(keys, sort=False).agg(
        n_series=("winner", "size"),
        wins_needed=("winner_wins", "max"),
        min_winner_wins=("winner_wins", "min"),
        draws=("draws", "sum"),
    )
    fmt = fmt.reset_index()
    fmt["round"] = fmt["round_key"].replace(-1.0, np.nan)
    # Every series in a round should reach the same count; if not the round mixes formats
    # (or a series was abandoned), which is flagged rather than guessed.
    fmt["consistent"] = fmt["wins_needed"].eq(fmt["min_winner_wins"])
    return fmt.drop(columns=["round_key"])


def top_places(series: pd.DataFrame) -> pd.DataFrame:
    """Places 1-4 from the main bracket's final and bronze series.

    Without a bronze series both semifinal losers share place 3 (``place_hi`` = 4).
    Only tournaments whose main bracket has a decided final are returned.
    """
    main = series.loc[series["bracket"].eq("") & series["decided"]]
    # The main bracket is the one in the last playoff stage holding a final.
    finals = main.loc[main["round"].eq(1.0)]
    last_stage = finals.sort_values(["stage_seq", "date"]).groupby("tournament_uid").tail(1)
    rows = []
    stage_of = dict(zip(last_stage["tournament_uid"], last_stage["stage_uid"]))
    for t, final in last_stage.set_index("tournament_uid").iterrows():
        rows.append((t, final["winner"], 1, 1))
        rows.append((t, final["loser"], 2, 2))
    places = pd.DataFrame(rows, columns=["tournament_uid", "player", "place", "place_hi"])
    in_stage = main.loc[main["tournament_uid"].map(stage_of).eq(main["stage_uid"])]
    bronze = in_stage.loc[in_stage["round"].eq(0.9)].groupby("tournament_uid").head(1)
    semis = in_stage.loc[in_stage["round"].eq(0.5)]
    extra = []
    for b in bronze.itertuples():
        extra.append((b.tournament_uid, int(b.winner), 3, 3))
        extra.append((b.tournament_uid, int(b.loser), 4, 4))
    # Without a decided bronze series both semifinal losers share places 3-4.
    for t, grp in semis.loc[~semis["tournament_uid"].isin(bronze["tournament_uid"])].groupby("tournament_uid"):
        for loser in grp["loser"]:
            extra.append((t, int(loser), 3, 4))
    if extra:
        places = pd.concat([places, pd.DataFrame(extra, columns=places.columns)], ignore_index=True)
    return places.sort_values(["tournament_uid", "place"]).reset_index(drop=True)


def stage_summary(games: pd.DataFrame) -> pd.DataFrame:
    """One row per stage: kind, players and the source's stage name."""
    games = structural(games)
    long = pd.concat(
        [
            games[["tournament_uid", "stage_uid", "stage_seq", "stage_kind", "p1"]].rename(columns={"p1": "p"}),
            games[["tournament_uid", "stage_uid", "stage_seq", "stage_kind", "p2"]].rename(columns={"p2": "p"}),
        ]
    )
    return (
        long.groupby(["tournament_uid", "stage_uid"])
        .agg(stage_seq=("stage_seq", "first"), stage_kind=("stage_kind", "first"), n_players=("p", "nunique"))
        .reset_index()
    )
