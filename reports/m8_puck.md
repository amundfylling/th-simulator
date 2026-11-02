# What the h2h.puck.no ratings add (game level)

Ideas from https://h2h.puck.no/ratings, scored on our data like the M7 hypotheses. Everything is
fitted on 2015-2021 (rating settings, blend weight, head-to-head coefficient, regulation model);
`select` = 2022-2024, `final` = 2025-2026 (untouched until the end). 3-way regulation log loss, Δ vs
today's Kalman model (negative = better), SE clustered by tournament. `pairs met 5+` = games where
the two players had met at least 5 times before.

| candidate | select log loss | Δ ×1000 (±SE) | Δ pairs met 5+ | Δ playoff | final log loss | Δ ×1000 (±SE) | Δ pairs met 5+ | Δ playoff |
|---|---|---|---|---|---|---|---|---|
| today: Kalman | 0.7885 | +0.00 (0.00) | +0.00 | +0.00 | 0.7954 | +0.00 (0.00) | +0.00 | +0.00 |
| P1 h2h.puck.no rating, published settings, raw gap | 0.7854 | -3.13 (0.27) | -3.36 | -3.33 | 0.7923 | -3.16 (0.31) | -3.95 | -2.57 |
| P1+P2 h2h.puck.no rating + forecast, published settings | 0.7851 | -3.40 (0.25) | -3.42 | -3.13 | 0.7920 | -3.42 (0.28) | -3.96 | -2.39 |
| P1+P2 same, tuned on our 2015-2021 games | 0.7836 | -4.89 (0.40) | -7.66 | -5.15 | 0.7896 | -5.84 (0.50) | -8.17 | -4.00 |
| P1 tuned, rating uncertainty integrated instead of g(RD) | 0.7863 | -2.17 (0.57) | -7.58 | -5.24 | 0.7917 | -3.74 (0.63) | -8.19 | -3.47 |
| P3 head-to-head edge on today's Kalman | 0.7869 | -1.60 (0.11) | -2.35 | -2.30 | 0.7935 | -1.95 (0.16) | -2.99 | -2.64 |
| H11 (M7) Kalman + results-only Glicko, for reference | 0.7871 | -1.41 (0.13) | -2.33 | -1.69 | 0.7934 | -2.05 (0.14) | -2.64 | -1.76 |
| Composite rating (weight 0.7 on tuned Glicko-MoV, sd ×0.65) | 0.7834 | -5.09 (0.34) | -7.35 | -5.53 | 0.7897 | -5.75 (0.39) | -7.76 | -4.57 |
| Composite + head-to-head edge (shipped) | 0.7824 | -6.10 (0.35) | -8.83 | -6.91 | 0.7884 | -7.07 (0.41) | -9.80 | -6.39 |

Games: select 271,127 (162,830 pairs met 5+, 65,855 playoff), final 178,095 (115,226 pairs met 5+, 41,089 playoff).

For scale: switching from Elo to the Kalman ratings was worth about −13 on this metric; the best M7
hypothesis (H11) about −1.4.

## Fitted settings

Glicko-MoV tuned on 2015-2021 (tuning log loss 0.7822 with the published settings → 0.7750):

```json
{
 "tau": 3.75,
 "sigma0": 0.28125,
 "phi0": 700.0,
 "mov_k": 0.8,
 "week_days": 3.36,
 "gd_cap": 10.0,
 "prior_mean": 0.0
}
```

Blend weight on the Glicko-MoV rating: **0.7** (scale 1.570 goals per 173.7 rating points). Blended sd multiplier: **0.65** (Glicko's RD overstates the spread). Tuning log loss by weight/sd multiplier: 0.0/0.3: 0.77751, 0.0/0.4: 0.77751, 0.0/0.5: 0.77750, 0.0/0.65: 0.77749, 0.0/0.8: 0.77748, 0.0/1.0: 0.77748, 0.5/0.3: 0.77283, 0.5/0.4: 0.77273, 0.5/0.5: 0.77266, 0.5/0.65: 0.77263, 0.5/0.8: 0.77274, 0.5/1.0: 0.77311, 0.6/0.3: 0.77266, 0.6/0.4: 0.77251, 0.6/0.5: 0.77241, 0.6/0.65: 0.77238, 0.6/0.8: 0.77253, 0.6/1.0: 0.77305, 0.7/0.3: 0.77272, 0.7/0.4: 0.77252, 0.7/0.5: 0.77238, 0.7/0.65: 0.77234, 0.7/0.8: 0.77256, 0.7/1.0: 0.77324, 0.8/0.3: 0.77301, 0.8/0.4: 0.77275, 0.8/0.5: 0.77257, 0.8/0.65: 0.77253, 0.8/0.8: 0.77281, 0.8/1.0: 0.77368, 1.0/0.3: 0.77424, 1.0/0.4: 0.77382, 1.0/0.5: 0.77355, 1.0/0.65: 0.77352, 1.0/0.8: 0.77397, 1.0/1.0: 0.77529.

Head-to-head edge: prior 20 meetings, coefficient 2.541 goals per unit of edge. Tuning log loss by prior: 10.0: 0.77174, 20.0: 0.77173, 40.0: 0.77174.
