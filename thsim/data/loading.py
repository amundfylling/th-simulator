"""Load, normalise and de-duplicate the game sources into one tidy games table.

Overtime parsing, walkover detection and the cross-source dedupe follow
``scripts/build_h2h.py`` in amundfylling/table-hockey-h2h-cloudflare.

Unified columns (one row per game):

    source            "scorpion" | "bordshockey"
    tournament_uid    int64, unique across sources
    stage_uid         int64, unique across sources
    tournament_id, stage_id   ids as given by the source
    tournament_name, stage_name
    stage_seq         order of the stage inside the tournament
    stage_kind        "rr" (round-robin / league) | "po" (playoff series)
    bracket           playoff bracket label ("A", "B", ... ; "" for the main one)
    round             rr: round number; po: 1=final, 0.9=bronze, 0.5=semi, 0.25=QF, ...
    game_no           game number inside a playoff series
    date              datetime64
    p1, p2, g1, g2    players and final goals (OT goal included)
    ot                golden-goal overtime was played
    walkover          game was not played (excluded from fitting, kept for structure)
    team_event        game was played inside a team tournament
    level             Scorpion tournament level (string, may be missing)
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from thsim.data.sources import fetch

BORDSHOCKEY_UID_OFFSET = 10**10
SYNTHETIC_PLAYER_ID_MIN = 1_000_000

TRUE_OVERTIME = {"1", "1.0", "true", "yes", "y", "ot", "overtime", "over time", "sudden death"}
TRUE_WALKOVER = {"1", "1.0", "true", "yes", "y", "walkover", "wo", "w/o"}

# Playoff round words (Swedish and English) -> Scorpion-style round fraction.
ROUND_WORDS = [
    (re.compile(r"brons|bronze|3rd place|match om 3", re.I), 0.9),
    (re.compile(r"semi", re.I), 0.5),
    (re.compile(r"kvarts|quarter", re.I), 0.25),
    (re.compile(r"\b8-dels|1/8|round of 16|eighth", re.I), 0.125),
    (re.compile(r"\b16-dels|1/16|round of 32", re.I), 0.0625),
    (re.compile(r"\b32-dels|1/32|round of 64", re.I), 0.03125),
    (re.compile(r"\b64-dels|1/64", re.I), 0.015625),
    (re.compile(r"final", re.I), 1.0),
]
BRACKET_RE = re.compile(r"^\s*(?:play-?off|slutspel)?\s*([A-H])[- ]?(?:slutspel|play-?off)?\b", re.I)


def to_int(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    if pd.api.types.is_float_dtype(numeric):
        numeric = numeric.where(numeric.isna() | (numeric == numeric.round(0)))
    return numeric.astype("Int64")


def _flag(series: pd.Series | None, true_values: set[str], index: pd.Index) -> pd.Series:
    if series is None:
        return pd.Series(False, index=index)
    text = series.astype("object").where(series.notna(), "").astype(str).str.strip().str.casefold()
    return text.isin(true_values)


def normalize_dedupe_text(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)) or value is pd.NA:
        return ""
    text = unicodedata.normalize("NFKD", str(value).strip())
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).casefold()
    return re.sub(r"[\W_]+", " ", text, flags=re.UNICODE).strip()


def playoff_round_from_name(name: str) -> float:
    for pattern, value in ROUND_WORDS:
        if pattern.search(name or ""):
            return value
    return np.nan


def bracket_from_name(name: str) -> str:
    match = BRACKET_RE.match(name or "")
    if match and match.group(1).upper() != "A":
        return match.group(1).upper()
    return ""


def _finish(df: pd.DataFrame) -> pd.DataFrame:
    """Validate ids/scores and cast the unified columns."""
    for col in ["p1", "p2", "g1", "g2"]:
        df[col] = to_int(df[col])
    valid = (
        df[["p1", "p2", "g1", "g2"]].notna().all(axis=1)
        & df["p1"].gt(0)
        & df["p2"].gt(0)
        & df["p1"].ne(df["p2"])
        & df["g1"].ge(0)
        & df["g2"].ge(0)
    ).fillna(False)
    df = df.loc[valid].copy()
    for col in ["p1", "p2", "g1", "g2"]:
        df[col] = df[col].astype("int64")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.loc[df["date"].notna()].copy()
    for col in ["stage_seq", "game_no"]:
        df[col] = to_int(df[col]).fillna(0).astype("int64")
    df["round"] = pd.to_numeric(df["round"], errors="coerce").astype("float64")
    for col in ["tournament_name", "stage_name", "bracket"]:
        df[col] = df[col].astype("object").where(df[col].notna(), "").astype(str)
    return df


def load_scorpion_games(path: Path | None = None) -> pd.DataFrame:
    path = path or fetch("scorpion_matches")
    raw = pd.read_parquet(path)
    df = pd.DataFrame(
        {
            "source": "scorpion",
            "tournament_id": raw["TournamentID"].astype("int64"),
            "stage_id": raw["StageID"].astype("int64"),
            "tournament_name": raw["TournamentName"],
            "stage_name": raw["Stage"],
            "stage_seq": raw["StageSequence"],
            "stage_kind": np.where(raw["Stage"].eq("Playoff"), "po", "rr"),
            "bracket": "",
            "round": raw["RoundNumber"],
            "game_no": raw["PlayoffGameNumber"],
            "date": raw["Date"],
            "p1": raw["Player1ID"],
            "p2": raw["Player2ID"],
            "g1": raw["GoalsPlayer1"],
            "g2": raw["GoalsPlayer2"],
            "match_id": raw["MatchID"].astype("int64"),
            "team_event": raw["TournamentType"].eq("Team"),
            "team_match_id": raw["TeamMatchID"],
        }
    )
    df["ot"] = _flag(raw["Overtime"], TRUE_OVERTIME, raw.index)
    df["walkover"] = _flag(raw["Walkover"], TRUE_WALKOVER, raw.index)
    df["tournament_uid"] = df["tournament_id"]
    df["stage_uid"] = df["stage_id"]
    return _finish(df)


def load_bordshockey_games(path: Path | None = None) -> pd.DataFrame:
    path = path or fetch("bordshockey_matches")
    raw = pd.read_parquet(path)
    stage_type = raw["StageType"].astype("object").fillna("").str.lower()
    is_po = stage_type.eq("playoff")
    names = raw["Stage"].astype("object").fillna("").astype(str)
    df = pd.DataFrame(
        {
            "source": "bordshockey",
            "tournament_id": to_int(raw["TournamentID"]),
            "stage_id": to_int(raw["StageID"]),
            "tournament_name": raw["TournamentName"].astype("object"),
            "stage_name": names,
            "stage_seq": raw["StageSequence"],
            "stage_kind": np.where(is_po, "po", "rr"),
            "bracket": np.where(is_po, names.map(bracket_from_name), ""),
            "round": np.where(is_po, names.map(playoff_round_from_name), raw["RoundNumber"].astype("float64")),
            "game_no": raw["PlayoffGameNumber"],
            "date": raw["Date"].astype("object"),
            "p1": raw["Player1ID"],
            "p2": raw["Player2ID"],
            "g1": raw["GoalsPlayer1"],
            "g2": raw["GoalsPlayer2"],
            "match_id": -1,
            "team_event": False,
            "team_match_id": np.nan,
        }
    )
    df["ot"] = _flag(raw["Overtime"], TRUE_OVERTIME, raw.index).to_numpy()
    walkover = raw["Walkover"] if "Walkover" in raw else None
    df["walkover"] = _flag(walkover, TRUE_WALKOVER, raw.index).to_numpy()
    df = df.loc[df["tournament_id"].notna() & df["stage_id"].notna()].copy()
    df["tournament_id"] = df["tournament_id"].astype("int64")
    df["stage_id"] = df["stage_id"].astype("int64")
    df["tournament_uid"] = df["tournament_id"] + BORDSHOCKEY_UID_OFFSET
    # A bordshockey StageID can hold several brackets (A/B/C playoffs).
    df["stage_uid"] = df["stage_id"] + BORDSHOCKEY_UID_OFFSET
    return _finish(df)


# --- cross-source dedupe (from build_h2h.py) -----------------------------------

EXACT_SIGNATURE = ["id1", "id2", "date", "stage_seq", "rr_round", "game_no", "goals_id1", "goals_id2", "ot"]
LOOSE_SIGNATURE = ["id1", "id2", "date", "goals_id1", "goals_id2", "ot"]
MIN_EXACT_MATCHES = 3
MIN_LOOSE_MATCHES = 10


def _signature_frame(games: pd.DataFrame) -> pd.DataFrame:
    swap = games["p1"] > games["p2"]
    return pd.DataFrame(
        {
            "id1": np.where(swap, games["p2"], games["p1"]),
            "id2": np.where(swap, games["p1"], games["p2"]),
            "date": games["date"].to_numpy(),
            "stage_seq": games["stage_seq"].to_numpy(),
            # Playoff round numbering differs between the sources; only rr rounds compare.
            "rr_round": np.where(games["stage_kind"].eq("rr"), games["round"].fillna(-1), -1.0),
            "game_no": games["game_no"].to_numpy(),
            "goals_id1": np.where(swap, games["g2"], games["g1"]),
            "goals_id2": np.where(swap, games["g1"], games["g2"]),
            "ot": games["ot"].to_numpy(),
        },
        index=games.index,
    )


def _crosswalk(sig, is_b, tids, names, columns, min_matches, use_names):
    hashes = pd.util.hash_pandas_object(sig[columns], index=False)
    b_hashes = set(hashes[is_b.to_numpy()].tolist())
    primary = sig.loc[(~is_b) & hashes.isin(b_hashes).to_numpy(), columns].copy()
    primary["_pi"] = primary.index
    primary["_pt"] = tids.loc[primary.index].to_numpy()
    primary["_pn"] = names.loc[primary.index].to_numpy()
    second = sig.loc[is_b, columns].copy()
    second["_bi"] = second.index
    second["_bt"] = tids.loc[second.index].to_numpy()
    second["_bn"] = names.loc[second.index].to_numpy()
    overlaps = primary.merge(second, on=columns, how="inner", sort=False)
    if overlaps.empty:
        return set(), []
    pair = ["_pt", "_bt"]
    evidence = overlaps.drop_duplicates([*pair, *columns]).groupby(pair).size()
    candidates = {k for k, v in evidence.items() if v >= min_matches}
    if use_names:
        same = overlaps["_pn"].map(normalize_dedupe_text)
        same = same.ne("") & same.eq(overlaps["_bn"].map(normalize_dedupe_text))
        candidates |= set(map(tuple, overlaps.loc[same, pair].drop_duplicates().to_numpy().tolist()))
    p_count: dict[int, int] = {}
    b_count: dict[int, int] = {}
    for p, b in candidates:
        p_count[p] = p_count.get(p, 0) + 1
        b_count[b] = b_count.get(b, 0) + 1
    accepted = {(p, b) for p, b in candidates if p_count[p] == 1 and b_count[b] == 1}
    keys = list(zip(overlaps["_pt"], overlaps["_bt"]))
    overlaps = overlaps.loc[[k in accepted for k in keys]]
    groups = []
    for (pt, bt, *_), grp in overlaps.groupby([*pair, *columns], sort=False):
        groups.append(((pt, bt), list(dict.fromkeys(grp["_pi"])), list(dict.fromkeys(grp["_bi"]))))
    return accepted, groups


def deduplicate_sources(games: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, int]]:
    """Drop Scorpion copies of games that bordshockey.net also has.

    Returns the de-duplicated games and the accepted tournament crosswalk
    {scorpion tournament_uid: bordshockey tournament_uid}. Like build_h2h.py the
    bordshockey copy is kept and its missing level is backfilled from Scorpion.
    """
    is_b = games["source"].eq("bordshockey")
    if not is_b.any() or is_b.all():
        return games, {}
    sig = _signature_frame(games)
    tids = games["tournament_uid"]
    names = games["tournament_name"]
    drop: list[int] = []
    keep: list[int] = []
    p2b: dict[int, int] = {}
    b2p: dict[int, int] = {}
    for columns, min_matches, use_names in [
        (EXACT_SIGNATURE, MIN_EXACT_MATCHES, True),
        (LOOSE_SIGNATURE, MIN_LOOSE_MATCHES, False),
    ]:
        remaining = games.index.difference(drop + keep)
        accepted, groups = _crosswalk(
            sig.loc[remaining], is_b.loc[remaining], tids, names, columns, min_matches, use_names
        )
        accepted = {
            (p, b) for p, b in accepted if p2b.get(p, b) == b and b2p.get(b, p) == p
        }
        for p, b in accepted:
            p2b[p] = b
            b2p[b] = p
        for pair, p_idx, b_idx in groups:
            if pair not in accepted:
                continue
            n = min(len(p_idx), len(b_idx))
            keep.extend(b_idx[:n])
            drop.extend(p_idx[:n])
    games = games.copy()
    if drop and "level" in games:
        dropped_levels = games.loc[drop, "level"].to_numpy()
        kept_levels = games.loc[keep, "level"]
        fill = kept_levels.isna().to_numpy() & pd.notna(dropped_levels)
        games.loc[np.asarray(keep)[fill], "level"] = dropped_levels[fill]
    return games.drop(index=drop), {int(p): int(b) for p, b in p2b.items()}


# --- metadata -------------------------------------------------------------------


def load_tournaments(path: Path | None = None) -> pd.DataFrame:
    path = path or fetch("scorpion_tournaments")
    meta = pd.read_csv(path, encoding="utf-8-sig")
    meta["TournamentID"] = to_int(meta["TournamentID"])
    meta = meta.loc[meta["TournamentID"].notna()].drop_duplicates("TournamentID", keep="last")
    level = pd.to_numeric(meta["Level"], errors="coerce")
    return pd.DataFrame(
        {
            "tournament_uid": meta["TournamentID"].astype("int64").to_numpy(),
            "name": meta["Name"].to_numpy(),
            "type": meta["Type"].to_numpy(),
            "level": level.to_numpy(),
            "country": meta["Country"].to_numpy(),
            "date": pd.to_datetime(meta["Date"], errors="coerce").to_numpy(),
            "participants": pd.to_numeric(meta["Participants"], errors="coerce").to_numpy(),
            "detail_url": meta["DetailURL"].to_numpy(),
        }
    )


def load_players(path: Path | None = None) -> pd.DataFrame:
    path = path or fetch("scorpion_players")
    players = pd.read_csv(path, encoding="utf-8-sig")
    players["PlayerID"] = to_int(players["PlayerID"])
    players = players.loc[players["PlayerID"].notna()].drop_duplicates("PlayerID", keep="last")
    return pd.DataFrame(
        {
            "player_id": players["PlayerID"].astype("int64").to_numpy(),
            "name": players["Name"].fillna("").astype(str).str.strip().to_numpy(),
            "ithf_id": to_int(players["RankingID"]).to_numpy(),
            "country": players["Country"].to_numpy(),
            "birth_date": pd.to_datetime(players["DateOfBirth"], format="%d.%m.%Y", errors="coerce").to_numpy(),
        }
    )


RANKING_HEADER = "Rank ID_Player Player Club Nation Points Player_Value"
RANKING_ROW_RE = re.compile(
    r"(?:^|\s)(\d+)\s+(\d+)\s+(.+?)\s+([A-Z]{3})\s+(\d+)\s+(\d+)(?=\s+\d+\s+\d+\s+|$)",
    re.DOTALL,
)


def parse_ranking(text: str) -> pd.DataFrame:
    """Parse the ITHF world ranking text file (same regex as build_h2h.py)."""
    compact = re.sub(r"\s+", " ", text).strip()
    if RANKING_HEADER in compact:
        compact = compact.split(RANKING_HEADER, 1)[1].strip()
    rows = [
        (int(rank), int(pid), name_club.strip(), nation, int(points), int(value))
        for rank, pid, name_club, nation, points, value in RANKING_ROW_RE.findall(compact)
    ]
    return pd.DataFrame(rows, columns=["world_rank", "ithf_id", "name_club", "nation", "points", "player_value"])


def load_ranking(path: Path | None = None) -> pd.DataFrame | None:
    path = path or fetch("ithf_ranking", required=False)
    if path is None:
        return None
    return parse_ranking(path.read_text(encoding="utf-8-sig", errors="replace"))


# --- public entry point -----------------------------------------------------------

SORT_KEYS = ["date", "tournament_uid", "stage_seq", "stage_uid", "round", "game_no", "match_id"]


def load_games(include_bordshockey: bool = True, dedupe: bool = True) -> pd.DataFrame:
    """All games from both sources, chronologically sorted, with a stable game index."""
    frames = [load_scorpion_games()]
    if include_bordshockey:
        frames.append(load_bordshockey_games())
    games = pd.concat(frames, ignore_index=True)
    tournaments = load_tournaments()
    games["level"] = games["tournament_uid"].map(tournaments.set_index("tournament_uid")["level"])
    crosswalk: dict[int, int] = {}
    if dedupe:
        games, crosswalk = deduplicate_sources(games)
    games = games.sort_values(SORT_KEYS, kind="stable").reset_index(drop=True)
    games.attrs["crosswalk"] = crosswalk
    return games
