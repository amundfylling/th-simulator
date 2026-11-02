import pandas as pd

from conftest import make_games
from thsim.data.loading import (
    _flag, bracket_from_name, deduplicate_sources, parse_ranking, playoff_round_from_name,
    TRUE_OVERTIME,
)


def test_overtime_parsing():
    s = pd.Series(["Yes", "no", None, "OT", "1.0", "weird"])
    assert _flag(s, TRUE_OVERTIME, s.index).tolist() == [True, False, False, True, True, False]


def test_playoff_round_and_bracket_names():
    assert playoff_round_from_name("A-slutspel Semifinal") == 0.5
    assert playoff_round_from_name("Slutspel Bronsmatch") == 0.9
    assert playoff_round_from_name("Playoff 8-delsfinal") == 0.125
    assert playoff_round_from_name("Slutspel Final") == 1.0
    assert bracket_from_name("B-slutspel Final") == "B"
    assert bracket_from_name("A-slutspel Final") == ""
    assert bracket_from_name("Slutspel Final") == ""


def test_dedupe_keeps_bordshockey_copy_of_overlapping_tournament():
    rows = [(1, 2, 3, 1), (3, 4, 2, 2), (1, 3, 0, 1), (2, 4, 5, 0)]
    scorpion = make_games(rows, source="scorpion", tournament_uid=5, tournament_name="Cup", level=3)
    swapped = [(b, a, gb, ga) for a, b, ga, gb in rows[:3]]
    bord = make_games(swapped, source="bordshockey", tournament_uid=10**10 + 7, tournament_name="Other", level=None)
    unrelated = make_games([(1, 2, 3, 1)], source="scorpion", tournament_uid=6, date=pd.Timestamp("2021-01-01"))
    games = pd.concat([scorpion, bord, unrelated], ignore_index=True)
    out, crosswalk = deduplicate_sources(games)
    assert crosswalk == {5: 10**10 + 7}
    assert len(out) == len(games) - 3
    assert (out["source"] == "scorpion").sum() == 2  # the non-overlapping game and the unrelated one
    assert out.loc[out["source"].eq("bordshockey"), "level"].eq(3).all()


def test_parse_ranking():
    text = "ITHF ranking up to 1.10.2026\nRank ID_Player Player Club Nation Points Player_Value\n1 123 Some Player Club X SWE 9000 5000\n2 456 Another One RUS 8000 4000\n"
    r = parse_ranking(text)
    assert r["ithf_id"].tolist() == [123, 456]
    assert r["nation"].tolist() == ["SWE", "RUS"]


def test_dedupe_leftovers_join_the_matched_tournament():
    rows = [(1, 2, 3, 1), (3, 4, 2, 2), (1, 3, 0, 1), (2, 4, 5, 0)]
    scorpion = make_games(rows, source="scorpion", tournament_uid=5, tournament_name="Cup")
    bord = make_games(rows[:3], source="bordshockey", tournament_uid=10**10 + 7, tournament_name="Cup")
    out, _ = deduplicate_sources(pd.concat([scorpion, bord], ignore_index=True))
    leftover = out.loc[out["source"].eq("scorpion")]
    assert len(leftover) == 1 and leftover["dedupe_leftover"].all()
    assert leftover["tournament_uid"].eq(10**10 + 7).all()
