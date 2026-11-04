# M10: same-day form (game level)

Variants of the M9 LightGBM layer, all trained 2015-2020 and early-stopped on 2021. 3-way regulation log
loss; Δ ×1000 vs M9 as shipped (negative = better), SE clustered by tournament. Form is measured against
each player's rating before the event.

## select

| variant | all (n=271,127) | playoff after 3+ games each (n=62,835) | round-robin, 5+ games each (n=138,412) | first game of the event (n=13,722) |
|---|---|---|---|---|
| A no same-event features | 0.7742 (+2.04 ±0.23) | 0.8647 (+0.76 ±0.28) | 0.7640 (+2.43 ±0.33) | 0.6637 (+1.42 ±0.42) |
| B M9 as shipped | 0.7722 (+0.00 ±0.00) | 0.8640 (+0.00 ±0.00) | 0.7615 (+0.00 ±0.00) | 0.6623 (+0.00 ±0.00) |
| C + richer same-day form | 0.7710 (-1.19 ±0.13) | 0.8621 (-1.84 ±0.30) | 0.7602 (-1.31 ±0.20) | 0.6626 (+0.35 ±0.33) |

## final

| variant | all (n=178,095) | playoff after 3+ games each (n=39,482) | round-robin, 5+ games each (n=90,124) | first game of the event (n=9,723) |
|---|---|---|---|---|
| A no same-event features | 0.7819 (+2.50 ±0.26) | 0.8736 (+1.05 ±0.36) | 0.7744 (+3.00 ±0.36) | 0.6825 (+0.35 ±0.47) |
| B M9 as shipped | 0.7794 (+0.00 ±0.00) | 0.8725 (+0.00 ±0.00) | 0.7714 (+0.00 ±0.00) | 0.6821 (+0.00 ±0.00) |
| C + richer same-day form | 0.7784 (-1.00 ±0.16) | 0.8701 (-2.43 ±0.42) | 0.7708 (-0.66 ±0.21) | 0.6821 (+0.00 ±0.38) |

## Share of LightGBM gain from same-event features

- A no same-event features: none
- B M9 as shipped: ev_games_1 0.002, ev_games_2 0.002, ev_form_1 0.010, ev_form_2 0.011
- C + richer same-day form: ev_games_1 0.001, ev_games_2 0.001, ev_form_1 0.005, ev_form_2 0.005, day_score_1 0.002, day_score_2 0.002, day_goals_1 0.003, day_goals_2 0.003, day_last3_1 0.007, day_last3_2 0.008, day_drift_1 0.003, day_drift_2 0.003

## How big is same-day form, and who already uses it?

Playoff games in 2022-2026 where both players had 3+ games earlier in the event (n = 102,317),
grouped by how much better the in-form player did than their opponent that day (goals per game
vs pre-event expectation, shrunk). Score is win 1, draw 0.5, from the in-form player's view.

| form gap | games | actual score | pre-event rating says | rating updated through the day says |
|---|---|---|---|---|
| 0-0.25 | 24,339 | 0.496 | 0.491 | 0.499 |
| 0.25-0.5 | 21,810 | 0.492 | 0.471 | 0.495 |
| 0.5-1.0 | 31,166 | 0.485 | 0.444 | 0.489 |
| 1.0-1.5 | 15,370 | 0.477 | 0.402 | 0.477 |
| 1.5+ | 9,622 | 0.507 | 0.350 | 0.492 |

Same-day form is a strong signal: a player who is 1.5+ goals per game hotter than their opponent
scores 0.51 where the pre-event ratings expect 0.35. The game-level model already sees nearly all
of it, because the ratings update after every game of the day. The richer features above take a
further 1-2.4 ×1000 off playoff log loss.

The website's live mode does not: it fixes the games already played but simulates the rest with
pre-event strengths. That is where the idea has the most to give.
