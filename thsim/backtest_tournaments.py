"""Tournament-level walk-forward backtest.

1. ``build``: reconstruct historical individual tournaments as engine configs (format
   inference), attach every player's pre-event rating and the actual top places.
   Supported shapes: one round-robin league, groups followed by playoff brackets, and
   playoff-only events. The group draw is taken from the data; who advances is
   re-simulated with the default rules (2-1-0, head-to-head, goal difference, goals,
   cross-group ranking by position then per-game record), and ``rule_ok`` records
   whether those rules reproduce the real playoff field from the real group results.
2. The Node script ``web/scripts/backtest.ts`` simulates every event with the same
   engine the website uses.
3. ``score``: log loss of the actual winner, RPS over finishing places, calibration of
   P(win) and P(top 3), compared with plain Elo and base rates (paired per event).

Run: python -m thsim.backtest_tournaments build|score
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from thsim import metrics
from thsim.backtest_games import Context
from thsim.data.loading import load_games_cached, load_players
from thsim.fit_game_model import load_params
from thsim.ratings import models as M
from thsim.structure import assign_groups, playoff_series, structural

WORK = Path("data_cache/backtest")
WINDOWS = {"select": ("2022-01-01", "2024-12-31"), "final": ("2025-01-01", "2026-12-31")}
MIN_PLAYERS = 6


def _pre_event_ratings(games: pd.DataFrame, cols: dict[str, np.ndarray]) -> pd.DataFrame:
    """Each player's rating before their first game of each tournament."""
    long = pd.concat([
        pd.DataFrame({"t": games["tournament_uid"], "p": games["p1"], "i": games.index, **{k: v[0] for k, v in cols.items()}}),
        pd.DataFrame({"t": games["tournament_uid"], "p": games["p2"], "i": games.index, **{k: v[1] for k, v in cols.items()}}),
    ])
    return long.sort_values("i").drop_duplicates(["t", "p"]).set_index(["t", "p"])


def _table(games: pd.DataFrame, members: list[int]) -> list[int]:
    """Rank one group by the default chain: points 2-1-0, h2h among tied, gd, gf."""
    pts = {p: 0 for p in members}
    gd = {p: 0 for p in members}
    gf = {p: 0 for p in members}
    played = {p: 0 for p in members}
    for r in games.itertuples():
        for a, b, ga, gb in ((r.p1, r.p2, r.g1, r.g2), (r.p2, r.p1, r.g2, r.g1)):
            pts[a] += 2 if ga > gb else 1 if ga == gb else 0
            gd[a] += ga - gb
            gf[a] += ga
            played[a] += 1
    h2h = {p: 0 for p in members}
    by_pts: dict[int, list[int]] = {}
    for p in members:
        by_pts.setdefault(pts[p], []).append(p)
    for tied in by_pts.values():
        if len(tied) < 2:
            continue
        sub = games.loc[games["p1"].isin(tied) & games["p2"].isin(tied)]
        for r in sub.itertuples():
            h2h[r.p1] += 2 if r.g1 > r.g2 else 1 if r.g1 == r.g2 else 0
            h2h[r.p2] += 2 if r.g2 > r.g1 else 1 if r.g1 == r.g2 else 0
    order = sorted(members, key=lambda p: (-pts[p], -h2h[p], -gd[p], -gf[p]))
    return order, {p: (pts[p] / max(1, played[p]), gd[p] / max(1, played[p]), gf[p] / max(1, played[p])) for p in members}


def _cross_rank(group_orders: list[list[int]], per_game: dict) -> list[int]:
    ranking = []
    for pos in range(max(len(o) for o in group_orders)):
        tier = [o[pos] for o in group_orders if pos < len(o)]
        tier.sort(key=lambda p: tuple(-x for x in per_game[p]))
        ranking.extend(tier)
    return ranking


def _ko_stage(series: pd.DataFrame) -> dict | None:
    """Main-bracket config bits and actual top places from one bracket's series."""
    s = series.loc[series["decided"] & ~series["walkover"]]
    final = s.loc[s["round"].eq(1.0)]
    if len(final) != 1:
        return None
    rounds = sorted([r for r in s["round"].dropna().unique() if r < 0.9 or r == 1.0])
    wins = [int(s.loc[s["round"].eq(r), "winner_wins"].max()) for r in rounds]
    bronze = s.loc[s["round"].eq(0.9)]
    f = final.iloc[0]
    places = {int(f["winner"]): 1, int(f["loser"]): 2}
    if len(bronze) == 1:
        places[int(bronze.iloc[0]["winner"])] = 3
        places[int(bronze.iloc[0]["loser"])] = 4
        shared = False
    else:
        semis = s.loc[s["round"].eq(0.5)]
        if len(semis) != 2:
            return None
        for p in semis["loser"]:
            places[int(p)] = 3
        shared = True
    entrants = sorted(set(series["a"]).union(series["b"]))
    bronze_wins = int(bronze.iloc[0]["winner_wins"]) if len(bronze) == 1 else None
    return {"wins": wins, "bronze": len(bronze) == 1, "bronze_wins": bronze_wins, "places": places, "shared_third": shared,
            "entrants": entrants}


def build(window: str) -> Path:
    lo, hi = WINDOWS[window]
    games = load_games_cached()
    ctx = Context(games, load_players())
    params = load_params()
    r = M.run("kalman", ctx.prep, params)
    elo = M.run("elo", ctx.prep, M.EloParams())
    cols = {
        "mu": (r["mu1"].to_numpy(), r["mu2"].to_numpy()),
        "var": (r["var1"].to_numpy(), r["var2"].to_numpy()),
        "elo": (elo["mu1"].to_numpy(), elo["mu2"].to_numpy()),
        "n": (ctx.n1, ctx.n2),
    }
    ratings = _pre_event_ratings(games, cols)
    # Plain Elo enters the same engine on the goal scale: match its logit slope to Kalman's.
    from thsim.models import ordered_logit

    fit_mask = ctx.rr & ctx.window(("2015-01-01", "2021-12-31")) & ctx.known
    idx = np.flatnonzero(fit_mask)[::4]
    beta_k, _ = ordered_logit.fit((r["mu1"] - r["mu2"]).to_numpy()[idx], ctx.y3[idx])
    beta_e, _ = ordered_logit.fit((elo["mu1"] - elo["mu2"]).to_numpy()[idx], ctx.y3[idx])
    elo_scale = beta_e / beta_k
    g = structural(games)
    g = g.loc[(g["date"] >= lo) & (g["date"] <= hi) & ~g["team_event"]]
    events = []
    counts = {"seen": 0, "unsupported": 0, "kept": 0, "rule_ok": 0}
    for t, tg in g.groupby("tournament_uid", sort=False):
        counts["seen"] += 1
        stages = tg.groupby("stage_seq")["stage_kind"].agg(lambda k: k.iloc[0])
        kinds = list(stages)
        players = sorted(set(tg["p1"]).union(tg["p2"]))
        if len(players) < MIN_PLAYERS or tg["walkover"].mean() > 0.05:
            counts["unsupported"] += 1
            continue
        ev = None
        if kinds.count("rr") == 1 and kinds[0] == "rr":
            ev = _rr_event(tg, kinds)
        elif all(k == "po" for k in kinds):
            ev = _po_event(tg)
        if ev is None:
            counts["unsupported"] += 1
            continue
        ev["players"] = []
        for p in ev.pop("field"):
            row = ratings.loc[(t, p)]
            ev["players"].append({"id": int(p), "name": str(p), "mu": float(row["mu"]), "sd": float(np.sqrt(row["var"])),
                                  "elo": float(row["elo"]), "n": int(row["n"])})
        ev |= {"tournament_uid": int(t), "date": str(tg["date"].min().date()), "source": tg["source"].iloc[0],
               "name": str(tg["tournament_name"].iloc[0]), "level": None if pd.isna(tg["level"].iloc[0]) else float(tg["level"].iloc[0])}
        counts["kept"] += 1
        counts["rule_ok"] += bool(ev["rule_ok"])
        events.append(ev)
    WORK.mkdir(parents=True, exist_ok=True)
    out = WORK / f"events_{window}.json"
    out.write_text(json.dumps({"window": window, "counts": counts, "elo_scale": elo_scale, "events": events}))
    print(counts, "->", out)
    return out


def _rr_event(tg: pd.DataFrame, kinds: list[str]) -> dict | None:
    rr = tg.loc[tg["stage_kind"].eq("rr") & ~tg["walkover"]].copy()
    rr["group"] = assign_groups(rr).to_numpy()
    groups = []
    orders = []
    per_game = {}
    for _, gg in rr.groupby("group"):
        members = sorted(set(gg["p1"]).union(gg["p2"]))
        pairs = gg.assign(lo=np.minimum(gg.p1, gg.p2), hi=np.maximum(gg.p1, gg.p2))[["lo", "hi"]].drop_duplicates()
        if len(pairs) != len(members) * (len(members) - 1) // 2:
            return None  # partial groups (Swiss / split leagues) are not reconstructed
        order, pg = _table(gg, members)
        groups.append([int(p) for p in members])
        orders.append(order)
        per_game |= pg
    legs = int(round(len(rr) / sum(len(m) * (len(m) - 1) // 2 for m in groups)))
    field = sorted(p for m in groups for p in m)
    group_stage = {"id": "groups", "type": "groups", "groups": len(groups), "legs": max(1, legs),
                   "groupsExplicit": groups, "source": [{"from": "entry"}], "places": 1}
    if len(kinds) == 1:
        if len(groups) != 1:
            return None
        actual = {int(p): k + 1 for k, p in enumerate(orders[0])}
        return {"kind": "league", "stages": [group_stage], "actual": actual, "known_places": len(field),
                "shared_third": False, "rule_ok": True, "field": field}
    po = tg.loc[tg["stage_kind"].eq("po")]
    if po.empty:
        return None
    series = playoff_series(po)
    # Main bracket: the one whose entrants did best in the groups.
    best = None
    ranking = _cross_rank(orders, per_game)
    rank_of = {p: k for k, p in enumerate(ranking)}
    for key, s in series.groupby(["stage_uid", "bracket"]):
        ko = _ko_stage(s)
        if ko is None:
            continue
        score = np.mean([rank_of.get(p, 999) for p in ko["entrants"]])
        if best is None or score < best[0]:
            best = (score, ko)
    if best is None:
        return None
    ko = best[1]
    k = len(ko["entrants"])
    if k < 2 or any(p not in rank_of for p in ko["entrants"]):
        return None
    rule_ok = set(ranking[:k]) == set(ko["entrants"])
    ko_stage = {"id": "ko", "type": "knockout", "winsNeeded": ko["wins"], **_bronze(ko),
                "source": [{"from": "groups", "ranks": [1, k]}], "places": 1}
    return {"kind": "groups_ko", "stages": [group_stage, ko_stage], "actual": ko["places"], "known_places": 4,
            "shared_third": ko["shared_third"], "rule_ok": rule_ok, "field": field, "ko_size": k}


def _bronze(ko: dict) -> dict:
    return {"bronze": True, "bronzeWins": ko["bronze_wins"]} if ko["bronze"] else {"bronze": False}


def _po_event(tg: pd.DataFrame) -> dict | None:
    series = playoff_series(tg)
    brackets = list(series.groupby(["stage_uid", "bracket"]))
    if len(brackets) != 1:
        return None
    ko = _ko_stage(brackets[0][1])
    if ko is None:
        return None
    field = ko["entrants"]
    stage = {"id": "ko", "type": "knockout", "winsNeeded": ko["wins"], **_bronze(ko),
             "source": [{"from": "entry"}], "places": 1}
    return {"kind": "ko", "stages": [stage], "actual": ko["places"], "known_places": 4,
            "shared_third": ko["shared_third"], "rule_ok": True, "field": field, "seed_by": "mu"}


# --- scoring --------------------------------------------------------------------------


def simulate(window: str, variants: list[str], runs: int, model: Path) -> Path:
    """Run the Node engine for the given variants and merge them into the predictions file."""
    out = WORK / f"pred_{window}.json"
    part = WORK / f"pred_{window}.part.json"
    cmd = ["npx", "tsx", "scripts/backtest.ts", str((WORK / f"events_{window}.json").resolve()), str(model.resolve()),
           str(part.resolve()), str(runs), *variants]
    subprocess.run(cmd, cwd="web", check=True)
    new = json.loads(part.read_text())
    if out.exists():
        old = json.loads(out.read_text())
        new["variants"] = old["variants"] | new["variants"]
    out.write_text(json.dumps(new))
    part.unlink()
    return out


def score(window: str, only_rule_ok: bool = False) -> dict:
    events = {e["tournament_uid"]: e for e in json.loads((WORK / f"events_{window}.json").read_text())["events"]}
    pred = json.loads((WORK / f"pred_{window}.json").read_text())
    rows = []
    calib = {}
    for variant, by_event in pred["variants"].items():
        for uid, res in by_event.items():
            ev = events[int(uid)]
            if only_rule_ok and not ev["rule_ok"]:
                continue
            ids = [p["id"] for p in ev["players"]]
            P = np.asarray(res["p"])  # players x places
            actual = {int(k): v for k, v in ev["actual"].items()}
            n = len(ids)
            winner = [i for i, p in enumerate(ids) if actual.get(p) == 1][0]
            k = ev["known_places"]
            # RPS over places 1..k plus "worse than k" (shared 3rd: 3 and 4 merged)
            cats = _categories(P, k, ev["shared_third"])
            outcome = np.array([_cat_of(actual.get(p), k, ev["shared_third"]) for p in ids])
            # Shared third: both semifinal losers count as top 3, so the engine's 3rd and 4th slots both qualify.
            top_k = 4 if ev["shared_third"] else 3
            top3 = P[:, :top_k].sum(axis=1)
            got3 = np.array([actual.get(p, 99) <= 3 for p in ids])
            uni = np.full(n, 1.0 / n)
            rows.append({
                "variant": variant, "uid": int(uid), "kind": ev["kind"], "n": n, "rule_ok": ev["rule_ok"],
                "winner_ll": -np.log(max(P[winner, 0], 1e-6)),
                "uniform_ll": -np.log(1.0 / n),
                "rps": metrics.rps(cats, outcome),
                "top3_brier": float(np.mean((top3 - got3) ** 2)),
                "uniform_top3_brier": float(np.mean((min(top_k, n) / n - got3) ** 2)),
                "p_winner": P[winner, 0],
                "fav_won": int(np.argmax(P[:, 0]) == winner),
            })
            c = calib.setdefault(variant, {"win": ([], []), "top3": ([], [])})
            c["win"][0].extend(P[:, 0]); c["win"][1].extend([actual.get(p) == 1 for p in ids])
            c["top3"][0].extend(top3); c["top3"][1].extend(got3)
    df = pd.DataFrame(rows)
    summary = df.groupby("variant").agg(events=("uid", "size"), winner_ll=("winner_ll", "mean"), uniform_ll=("uniform_ll", "mean"),
                                        rps=("rps", "mean"), top3_brier=("top3_brier", "mean"),
                                        uniform_top3_brier=("uniform_top3_brier", "mean"), fav_won=("fav_won", "mean"),
                                        mean_p_winner=("p_winner", "mean"))
    wide = df.pivot(index="uid", columns="variant", values="winner_ll")
    tests = {}
    for v in wide.columns:
        for w in wide.columns:
            if v < w:
                d = (wide[v] - wide[w]).dropna()
                tests[f"{v} vs {w}"] = {"mean_diff": float(d.mean()), "t_p": float(stats.ttest_1samp(d, 0).pvalue),
                                        "wilcoxon_p": float(stats.wilcoxon(d).pvalue), "n": int(len(d))}
    rel = {v: {k: metrics.reliability(np.array(a), np.array(b), bins=10) for k, (a, b) in c.items()} for v, c in calib.items()}
    return {"summary": summary, "tests": tests, "reliability": rel, "rows": df, "by_kind": df.groupby(["variant", "kind"])[["winner_ll", "uniform_ll", "rps"]].mean()}


def _categories(P: np.ndarray, k: int, shared: bool) -> np.ndarray:
    cols = [P[:, i] for i in range(min(k, P.shape[1]))]
    if shared and k == 4:
        cols = [P[:, 0], P[:, 1], P[:, 2] + P[:, 3]]
    rest = 1.0 - np.sum(cols, axis=0)
    return np.column_stack([*cols, np.clip(rest, 0, 1)])


def _cat_of(place, k, shared):
    if place is None:
        return 3 if (shared and k == 4) else k
    if shared and k == 4:
        return min(place, 3) - 1
    return place - 1 if place <= k else k


def report(window: str, res_all: dict, res_ok: dict, counts: dict, form_note: str) -> str:
    def md(df):
        df = df.reset_index()
        head = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
        return head + "\n".join("| " + " | ".join(f"{x:.4f}" if isinstance(x, float) else str(x) for x in r) + " |" for r in df.itertuples(index=False))

    tests = "\n".join(f"| {k} | {v['n']} | {v['mean_diff']:+.4f} | {v['t_p']:.3g} | {v['wilcoxon_p']:.3g} |" for k, v in res_all["tests"].items())
    rel = res_all["reliability"]
    best = res_all["summary"]["winner_ll"].idxmin()
    rel_md = "\n".join(f"| {a:.3f} | {b:.3f} | {n:,} |" for a, b, n in rel[best]["win"])
    rel3 = "\n".join(f"| {a:.3f} | {b:.3f} | {n:,} |" for a, b, n in rel[best]["top3"])
    return f"""# Milestone 5: tournament-level backtest ({window})

Events reconstructed: {counts}. Every event is simulated with the website's engine (Node), ratings
from games before each player's first game of the event. `rule_ok` = the default advancement
rules reproduce the real playoff field from the real group results.
{form_note}

## All reconstructed events

{md(res_all['summary'])}

`winner_ll`: mean -log P(actual winner wins). `rps`: ranked probability score over the known places
(1, 2, 3, 4 and "lower"; whole table for leagues). `top3_brier`: Brier score of P(top 3). Uniform =
every player equally likely.

### By event kind

{md(res_all['by_kind'])}

### Events where the rules reproduce who advanced

{md(res_ok['summary'])}

## Paired per-event test on winner log loss (negative = first variant better)

| comparison | events | mean diff | t-test p | Wilcoxon p |
|---|---|---|---|---|
{tests}

## Calibration ({best})

P(win):

| forecast | observed | player-events |
|---|---|---|
{rel_md}

P(top 3):

| forecast | observed | player-events |
|---|---|---|
{rel3}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["build", "simulate", "score", "all"])
    parser.add_argument("--window", default="select", choices=list(WINDOWS))
    parser.add_argument("--runs", type=int, default=2000)
    parser.add_argument("--variants", nargs="*", default=["kalman", "kalman_form0.15", "kalman_form0.3", "kalman_nounc", "elo"])
    parser.add_argument("--model", type=Path, default=WORK / "model_2021.json")
    args = parser.parse_args()
    if args.action in ("build", "all"):
        build(args.window)
    if args.action in ("simulate", "all"):
        simulate(args.window, args.variants, args.runs, args.model)
    if args.action in ("score", "all"):
        counts = json.loads((WORK / f"events_{args.window}.json").read_text())["counts"]
        res_all = score(args.window)
        res_ok = score(args.window, only_rule_ok=True)
        best = res_all["summary"]["winner_ll"].idxmin()
        note = f"Best variant by winner log loss: `{best}`."
        text = report(args.window, res_all, res_ok, counts, note)
        Path(f"reports/m5_tournaments_{args.window}.md").write_text(text)
        summary = res_all["summary"][["events", "winner_ll", "uniform_ll", "rps", "top3_brier", "fav_won"]]
        Path(f"reports/m5_tournaments_{args.window}.json").write_text(
            json.dumps({"window": WINDOWS[args.window], "variants": summary.round(4).to_dict(orient="index")}, indent=1))
        print(text)


if __name__ == "__main__":
    main()
