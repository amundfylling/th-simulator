# M9: ideas from a LightGBM experiment (game level)

Baseline is today's shipped model (composite rating + head-to-head edge, M8). Everything is fitted on
2015-2021 (LightGBM: trained 2015-2020, early-stopped on 2021); `select` = 2022-2024, `final` = 2025-2026.
Log loss is 3-way regulation (win / draw-or-OT / loss), Δ ×1000 vs today (negative = better), SE clustered
by tournament. `few` = games where a player had fewer than 30 earlier games. `fav won` = share of decided
games won by the side the model favoured; `top pick` = most likely of the three outcomes happened.

LightGBM is order-free: every training game appears from both sides, and predictions average both views,
so player 1 / player 2 order cannot leak the result.

## select (271,127 games, 29,803 with a newcomer, 65,855 playoff)

| model | log loss | Δ ×1000 (±SE) | Δ playoff | Δ few | fav won | top pick | draws pred / real (rr) |
|---|---|---|---|---|---|---|---|
| today (composite + head-to-head) | 0.7824 | +0.00 (0.00) | +0.00 | +0.00 | 76.14% | 66.73% | 0.126 / 0.129 |
| + per-player draw rate (draw cut) | 0.7820 | -0.46 (0.08) | +0.10 | -2.18 | 76.14% | 66.73% | 0.127 / 0.129 |
| + newcomer uncertainty | 0.7824 | -0.01 (0.01) | +0.01 | -0.16 | 76.14% | 66.73% | 0.125 / 0.129 |
| + recent form (EWMA residual, 30 games) | 0.7818 | -0.62 (0.09) | -0.39 | +0.25 | 76.15% | 66.75% | 0.126 / 0.129 |
| + all three | 0.7814 | -1.07 (0.13) | -0.27 | -2.07 | 76.15% | 66.75% | 0.127 / 0.129 |
| LightGBM layer (ratings + 30 features) | 0.7722 | -10.23 (0.59) | -7.58 | -36.86 | 76.51% | 67.08% | 0.125 / 0.129 |

## final (178,095 games, 13,740 with a newcomer, 41,089 playoff)

| model | log loss | Δ ×1000 (±SE) | Δ playoff | Δ few | fav won | top pick | draws pred / real (rr) |
|---|---|---|---|---|---|---|---|
| today (composite + head-to-head) | 0.7884 | +0.00 (0.00) | +0.00 | +0.00 | 75.84% | 66.36% | 0.127 / 0.131 |
| + per-player draw rate (draw cut) | 0.7879 | -0.48 (0.12) | +0.28 | -3.05 | 75.84% | 66.36% | 0.129 / 0.131 |
| + newcomer uncertainty | 0.7884 | -0.02 (0.01) | -0.01 | -0.32 | 75.84% | 66.36% | 0.127 / 0.131 |
| + recent form (EWMA residual, 30 games) | 0.7876 | -0.78 (0.10) | -0.11 | +0.35 | 75.90% | 66.42% | 0.127 / 0.131 |
| + all three | 0.7871 | -1.27 (0.15) | +0.15 | -3.01 | 75.89% | 66.42% | 0.129 / 0.131 |
| LightGBM layer (ratings + 30 features) | 0.7794 | -8.97 (0.65) | -7.39 | -38.00 | 76.12% | 66.64% | 0.127 / 0.131 |

## LightGBM feature importance (share of gain, top 20)

| feature | share |
|---|---|
| d | 0.720 |
| kal_2 | 0.036 |
| elo_1 | 0.035 |
| elo_2 | 0.032 |
| kal_1 | 0.031 |
| mu_1 | 0.012 |
| mu_2 | 0.011 |
| ev_form_2 | 0.011 |
| ev_form_1 | 0.010 |
| form_1 | 0.008 |
| games_2 | 0.008 |
| form_2 | 0.007 |
| games_1 | 0.007 |
| h2h | 0.007 |
| age_1 | 0.006 |
| age_2 | 0.006 |
| rd_2 | 0.006 |
| rd_1 | 0.006 |
| gl_2 | 0.005 |
| gl_1 | 0.005 |

Boosting rounds: 346.
