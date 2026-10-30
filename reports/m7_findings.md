# What would make the predictions better? Eleven hypotheses, tested walk-forward

Every hypothesis adds information that is known before the game (country pool offsets, age,
time since the last game, and so on) to the Kalman rating gap. Its coefficients are fitted on
2015–2021 together with the game model and scored on 2022–2024; 2025–2026 was scored once at
the end. Game-level details are in `m7_hypotheses.md`. The best candidates were then run
through the tournament engine on the M5 backtest events (`m7_tournaments_{select,final}.json`).

## Game level (3-way regulation log loss, Δ ×1000 vs today's model, negative = better)

| # | hypothesis | 2022–24 Δ (±SE) | 2025–26 Δ (±SE) | international games, 2022–24 | verdict |
|---|---|---|---|---|---|
| H11 | **Margins vs results**: some players pile up big wins over weak opponents, and a goal-difference rating overrates them. A results-only rating (Glicko-2) gets weight 0.16 | **−1.41 (0.13)** | **−2.05 (0.14)** | −1.60 | real, holds on the holdout |
| H1 | **Country pools drift**: online country offsets learned from cross-border games, scaled by how insular each player is | −0.12 (0.03) | −0.07 (0.03) | −0.60 | real but tiny |
| H6 | Juniors improve faster, veterans decline | −0.08 (0.03) | −0.01 (0.04) | −0.17 | did not hold on the holdout |
| H2 | Favourites over- or under-confident at large gaps | −0.05 (0.02) | −0.05 (0.02) | −0.06 | negligible |
| H5 | Fatigue late in an event | 0.00 | 0.00 | 0.00 | no effect |
| H10 | Momentum inside a playoff series | 0.00 (0.04) | +0.04 (0.05) | −0.03 | no effect |
| H7 | Rust after a long break | 0.00 | 0.00 | −0.01 | no effect (the wider uncertainty already covers it) |
| H9 | Home-country advantage | +0.02 (0.01) | +0.01 (0.01) | +0.20 | no effect (also +0.02 on events with a known host country) |
| H8 | Favourites more (or less) dominant at big events | +0.03 (0.01) | +0.02 (0.02) | −0.02 | no effect |
| H3 | Tighter or looser goal-difference cap in the ratings | +0.30 (0.07) | +0.43 (0.08) | +0.37 | worse; the tuned cap of 6 stays |
| H4 | Hot or cold form within an event | +0.36 (0.10) | +0.44 (0.11) | −0.60 | worse; the ratings already update game by game |
| | All helpful ones together (H11 + H1 + H6 + H2) | −1.65 (0.13) | −2.18 (0.16) | −2.50 | |

For scale: switching from Elo to the Kalman ratings was worth about −13 on the same metric.
H11 is real and about ten times larger than any other effect, but it is still small.

## Country pools and the Norway example

In cross-border games since 2019 (both players with 30+ games), the average goal-difference
residual is −0.10 goals per game for Norway, −0.11 for Germany and −0.32 for Slovakia, against
+0.13 for Ukraine and +0.09 for Latvia. In results (expected score), the gap almost vanishes:
Norway −0.001, Germany −0.010, Slovakia −0.028. Pools drift a little in margin, but enough
players cross borders that results-based strength stays aligned.

Magnus Klippen, 2019 onward: against Norwegians, expected score 0.817 and actual 0.881;
against foreigners, expected 0.667 and actual 0.709. He beats the model's expectation in both
pools. The model expects him to do worse abroad because the foreign opposition is stronger, and
he does a bit better than that expectation.

## Tournament level (2,000 runs per event, form shock 0.45)

| variant | 2022–24 winner log loss | 2022–24 top-3 Brier | 2025–26 winner log loss | 2025–26 top-3 Brier |
|---|---|---|---|---|
| today's model | 1.2076 | 0.1074 | **1.2924** | 0.1113 |
| + results-only blend (H11) | 1.2070 | **0.1067** | 1.3205 | 0.1114 |
| + H11 + country + age | **1.2062** | 0.1070 | 1.3135 | **0.1112** |

Paired t-tests on winner log loss: p = 0.90 and 0.21 for H11, and 0.78 and 0.29 for the combined version (2022–24 and 2025–26). Country offsets are frozen at each event's start here, so no entrant's rating sees a result from the event being forecast. The game-level gain does not carry
through to tournament outcomes. A whole tournament is decided by a handful of close games,
and a 0.2% better game forecast is lost in that noise.

## Recommendation

- **Ship nothing to the website for now.** None of the candidates improves tournament
  forecasts, and the blend makes the model harder to explain.
- Keep H11 in mind if ratings are ever shown as a ranking: a 16% results-only blend is the
  better game-level rating.
- Tournament calibration at the top end was fine in 2022–24 (forecasts of 0.85 won 93%) and
  overconfident in 2025–26 (0.85 won 75%, 61 player-events). That looks like holdout noise,
  so no shrinkage is added.
- Bigger levers lie outside the rating model: real formats for live events (exact draw,
  seeding and advancement rules, which the engine already supports) and live mode during an
  event, where games already played remove most of the uncertainty.
