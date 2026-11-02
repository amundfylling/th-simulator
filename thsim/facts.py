"""Reproduce the measured facts about the data as sanity checks.

Run: python -m thsim.facts [--out reports/m1_facts.md]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from thsim import metrics
from thsim.data.loading import load_games
from thsim.models import ordered_logit
from thsim.ratings.elo import run_elo
from thsim.structure import group_summary, playoff_series, round_formats, stage_summary

GAP_BINS = [0, 50, 100, 200, 300, 400, 600, np.inf]
GAP_LABELS = ["0-50", "50-100", "100-200", "200-300", "300-400", "400-600", "600+"]


def outcome_codes(g1: np.ndarray, g2: np.ndarray) -> np.ndarray:
    """0 = player 1 wins, 1 = draw, 2 = player 2 wins (regulation score)."""
    return np.where(g1 > g2, 0, np.where(g1 == g2, 1, 2))


def regulation_scores(games: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Score after regulation: an OT game was drawn at the loser's goal count."""
    lo = np.minimum(games["g1"], games["g2"]).to_numpy()
    g1 = np.where(games["ot"], lo, games["g1"])
    g2 = np.where(games["ot"], lo, games["g2"])
    return g1, g2


def compute(games: pd.DataFrame) -> dict:
    elo = run_elo(games)
    games = games.join(elo)
    games["gap"] = games["r1"] - games["r2"]
    games["abs_gap"] = games["gap"].abs()
    games["gap_bin"] = pd.cut(games["abs_gap"], GAP_BINS, labels=GAP_LABELS, right=False)
    played = games.loc[~games["walkover"]].copy()
    rg1, rg2 = regulation_scores(played)
    played["reg_draw"] = rg1 == rg2
    rr = played.loc[played["stage_kind"].eq("rr")]
    po = played.loc[played["stage_kind"].eq("po")]
    seasoned = (played["n1"] >= 30) & (played["n2"] >= 30)
    f: dict = {"n_games": len(games), "n_walkovers": int(games["walkover"].sum())}
    f["by_source"] = games["source"].value_counts().to_dict()

    # Draws in round-robin, overall and by Elo gap (both players seasoned).
    f["rr_draw_rate"] = rr["reg_draw"].mean()
    rr_s = rr.loc[seasoned.loc[rr.index]]
    f["rr_draw_by_gap"] = rr_s.groupby("gap_bin", observed=True)["reg_draw"].agg(["mean", "size"])

    # Playoff overtime and drawn-without-OT games.
    f["po_ot_rate"] = po["ot"].mean()
    ot = po.loc[po["ot"]]
    f["ot_one_goal_share"] = ((ot["g1"] - ot["g2"]).abs() == 1).mean()
    f["po_drawn_no_ot"] = ((po["g1"] == po["g2"]) & ~po["ot"]).mean()

    # Favourite win rate: regulation vs overtime playoff games.
    po_s = po.loc[seasoned.loc[po.index] & (po["g1"] != po["g2"])].copy()
    po_s["fav_won"] = np.sign(po_s["gap"]) == np.sign(po_s["g1"] - po_s["g2"])
    po_s = po_s.loc[po_s["abs_gap"] > 0]
    f["po_fav_by_gap"] = po_s.groupby(["gap_bin", "ot"], observed=True)["fav_won"].mean().unstack()

    # Goals.
    goals = np.concatenate([played["g1"], played["g2"]])
    f["goals_mean"], f["goals_var"] = goals.mean(), goals.var()
    reg = rr  # regulation-only scores for correlation and totals
    eq = reg.loc[seasoned.loc[reg.index] & (reg["abs_gap"] < 50)]
    f["goal_corr_equal"] = np.corrcoef(eq["g1"], eq["g2"])[0, 1]
    f["total_goals_by_gap"] = (
        rr_s.assign(total=rr_s["g1"] + rr_s["g2"]).groupby("gap_bin", observed=True)["total"].mean()
    )

    # Playoff series lengths.
    series = playoff_series(games)
    fmts = round_formats(series)
    f["wins_needed"] = fmts["wins_needed"].value_counts().sort_index()
    f["round_format_consistent"] = fmts["consistent"].mean()
    ok = series.loc[series["decided"] & ~series["walkover"]]
    f["n_series"] = len(ok)

    # Round-robin groups.
    groups = group_summary(games)
    f["n_groups"] = len(groups)
    f["groups_complete"] = groups["complete"].mean()
    f["group_median_size"] = groups["n_players"].median()
    f["groups_double"] = (groups["complete"] & (groups["legs"] >= 1.9)).mean()

    # Stages per tournament.
    stages = stage_summary(games)
    n_stage = stages.groupby("tournament_uid")["stage_seq"].nunique()
    f["multi_stage_share"] = (n_stage > 1).mean()
    f["max_stages"] = int(n_stage.max())
    f["stages_dist"] = n_stage.value_counts().sort_index()

    # Baseline: Elo + ordered logit, fit 2015-2021, evaluate 2022-2026, both seasoned.
    rr_s = rr_s.copy()
    rr_s["outcome"] = outcome_codes(*regulation_scores(rr_s))
    train = rr_s.loc[rr_s["date"].between("2015-01-01", "2021-12-31")]
    test = rr_s.loc[rr_s["date"] >= "2022-01-01"]
    beta, c = ordered_logit.fit(train["gap"].to_numpy(), train["outcome"].to_numpy())
    p = ordered_logit.probs(test["gap"].to_numpy(), beta, c)
    y = test["outcome"].to_numpy()
    draw = train["outcome"].eq(1).mean()
    base = np.tile([(1 - draw) / 2, draw, (1 - draw) / 2], (len(test), 1))
    f["baseline"] = {
        "beta": beta,
        "c": c,
        "n_train": len(train),
        "n_test": len(test),
        "elo_logloss": metrics.log_loss(p, y),
        "elo_brier": metrics.brier(p, y),
        "elo_rps": metrics.rps(p, y),
        "base_logloss": metrics.log_loss(base, y),
        "base_rps": metrics.rps(base, y),
    }
    return f


def _table(df: pd.DataFrame | pd.Series, fmt: str = "{:.3f}") -> str:
    df = df.to_frame() if isinstance(df, pd.Series) else df
    head = "| " + " | ".join([str(df.index.name or "")] + [str(c) for c in df.columns]) + " |"
    sep = "|" + "---|" * (len(df.columns) + 1)
    rows = []
    for idx, row in df.iterrows():
        cells = [fmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in row]
        rows.append("| " + " | ".join([str(idx)] + cells) + " |")
    return "\n".join([head, sep, *rows])


def report(f: dict) -> str:
    b = f["baseline"]
    by_gap_fav = f["po_fav_by_gap"].rename(columns={False: "regulation", True: "overtime"})
    return f"""# Milestone 1: data facts

Games after cleaning and cross-source dedupe: {f['n_games']:,} ({', '.join(f'{k} {v:,}' for k, v in f['by_source'].items())}), of which {f['n_walkovers']:,} walkovers (kept for structure, excluded from fitting).
Gap = pre-game Elo difference (K=24, K=64 decaying over a player's first 40 games). "Seasoned" = both players have 30+ prior games.

| Fact | Expected | Measured |
|---|---|---|
| Round-robin draw rate | 12.5% | {f['rr_draw_rate']:.1%} |
| RR draws, equal players (gap < 50) | ~18% | {f['rr_draw_by_gap']['mean'].iloc[0]:.1%} |
| RR draws, gap 600+ | < 2% | {f['rr_draw_by_gap']['mean'].iloc[-1]:.1%} |
| Playoff games to golden-goal OT | 13.2% | {f['po_ot_rate']:.1%} |
| OT games won by exactly one goal | 100% | {f['ot_one_goal_share']:.2%} |
| Playoff games drawn without OT | ~1% | {f['po_drawn_no_ot']:.2%} |
| Favourite wins, gap 200-300, regulation playoff | 83% | {by_gap_fav.loc['200-300', 'regulation']:.1%} |
| Favourite wins, gap 200-300, overtime | 67% | {by_gap_fav.loc['200-300', 'overtime']:.1%} |
| Goals per player: mean | 2.77 | {f['goals_mean']:.2f} |
| Goals per player: variance | 4.26 | {f['goals_var']:.2f} |
| Score correlation, equal players | -0.10 | {f['goal_corr_equal']:+.2f} |
| Total goals, gap 0-50 to 600+ | 5.1 to 6.9 | {f['total_goals_by_gap'].iloc[0]:.1f} to {f['total_goals_by_gap'].iloc[-1]:.1f} |
| RR groups complete | 89% | {f['groups_complete']:.1%} |
| Median group size | 9 | {f['group_median_size']:.0f} |
| Multi-stage tournaments | most | {f['multi_stage_share']:.1%} |
| Max stages in a tournament | 10 | {f['max_stages']} |
| Baseline Elo + ordered logit, log loss (3-way) | 0.791 | {b['elo_logloss']:.3f} |
| Base rates, log loss | 0.995 | {b['base_logloss']:.3f} |
| Baseline RPS | 0.170 | {b['elo_rps']:.3f} |

Baseline fit on {b['n_train']:,} seasoned round-robin games 2015-2021 (beta={b['beta']:.5f}, c={b['c']:.3f}), evaluated on {b['n_test']:,} games 2022-2026. Brier {b['elo_brier']:.3f}, base-rate RPS {b['base_rps']:.3f}.

## Round-robin draws by gap (seasoned players)

{_table(f['rr_draw_by_gap'])}

## Favourite win share in decided playoff games

{_table(by_gap_fav)}

## Total goals by gap (round-robin, seasoned)

{_table(f['total_goals_by_gap'])}

## Playoff series length (wins needed per round)

{_table(f['wins_needed'], '{:.0f}')}

{f['n_series']:,} decided series. {f['round_format_consistent']:.1%} of rounds have every series winner on the same win count (the rest mix formats or include abandoned series).

## Round-robin groups

{f['n_groups']:,} groups (stages split by connected components). {f['groups_complete']:.1%} complete, {f['groups_double']:.1%} complete double round-robins, median size {f['group_median_size']:.0f}.

## Stages per tournament

{_table(f['stages_dist'], '{:.0f}')}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("reports/m1_facts.md"))
    parser.add_argument("--scorpion-only", action="store_true")
    args = parser.parse_args()
    games = load_games(include_bordshockey=not args.scorpion_only)
    text = report(compute(games))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
