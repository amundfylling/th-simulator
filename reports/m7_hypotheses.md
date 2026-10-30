# Hypotheses for systematic prediction error

Each hypothesis adds pre-game information to the Kalman rating gap; coefficients are fitted on
2015-2021 together with the regulation model and scored on 2022-2024 (3-way regulation log loss,
all games with both players known). Δ is ×1000 vs the baseline (negative = better), SE clustered
by tournament. `intl` = players from different countries; `po` = playoff games.

`host known` = events with a known host country (Scorpion ids), the only games where H9 can act.

| hypothesis | chosen | coefficients (goals) | Δ all | ±SE | Δ intl | Δ po | Δ host known |
|---|---|---|---|---|---|---|---|
| Combined: H11 + H1 + H6 + H2 |  | +0.174, -0.015, +0.903, -0.031, -0.053, +0.215, -0.121 | -1.65 | 0.13 | -2.50 | -1.81 | -1.39 |
| H11 margins vs results (blowout specialists) |  | +0.163 | -1.41 | 0.13 | -1.60 | -1.69 | -1.14 |
| H1 country pools | {'lr': 0.003} | +0.004, +0.698 | -0.12 | 0.03 | -0.60 | -0.03 | -0.12 |
| H6 juniors improve, veterans decline |  | -0.040, -0.058 | -0.08 | 0.03 | -0.17 | -0.14 | -0.13 |
| H2 gap shape (fav over/under-confidence) |  | +0.219, -0.123 | -0.05 | 0.02 | -0.06 | +0.02 | -0.03 |
| H5 fatigue over the event |  | -0.008 | -0.00 | 0.00 | +0.00 | -0.00 | -0.00 |
| H10 momentum inside a series |  | +0.109 | -0.00 | 0.04 | -0.03 | -0.00 | -0.02 |
| baseline |  |  | +0.00 | 0.00 | +0.00 | +0.00 | +0.00 |
| H7 rust after a break |  | +0.037 | +0.00 | 0.00 | -0.01 | -0.00 | +0.00 |
| H9 home country advantage |  | +0.053 | +0.02 | 0.01 | +0.20 | -0.01 | +0.02 |
| H8 event level changes the gap |  | -0.014, -0.019 | +0.03 | 0.01 | -0.02 | -0.01 | +0.04 |
| H3 margin cap (blowouts inflate ratings) | {'cap': 8.0} |  | +0.30 | 0.07 | +0.37 | +0.45 | +0.36 |
| H4 form within the event | {'k': 2.0} | +0.110 | +0.36 | 0.10 | -0.60 | +0.59 | +0.34 |

Baseline log loss: 0.7885 over 271,127 games (31,545 international, 65,855 playoff).
