import pandas as pd


def make_games(rows, **defaults):
    """Build a unified games frame from (p1, p2, g1, g2) tuples plus column defaults."""
    base = dict(
        source="scorpion", tournament_uid=1, stage_uid=10, stage_seq=1, stage_kind="rr",
        stage_name="", bracket="", round=1.0, game_no=0, date=pd.Timestamp("2020-01-01"),
        ot=False, walkover=False, team_event=False, match_id=0, tournament_name="T",
    )
    base.update(defaults)
    out = []
    for r in rows:
        row = dict(base)
        if isinstance(r, dict):
            row.update(r)
        else:
            row.update(dict(zip(["p1", "p2", "g1", "g2"], r)))
        out.append(row)
    return pd.DataFrame(out)
