# M10: same-day form

Idea (Amund): a tournament is played in a day or two, so how a player is doing that day should
say something about their next games, the playoffs above all.

## 1. Is it real? Yes, and strongly

Playoff games 2022-2026 where both players had 3+ group games (`m10_form.md`): a player who is
1.5+ goals per game hotter than their opponent that day scores 0.51, where the pre-event
ratings expect 0.35.

## 2. Game-level predictions: already mostly used

The ratings update after every game, so by the playoffs they already expect 0.49 in that case.
Richer same-day features in the LightGBM layer (score and goals vs pre-event expectation, last
three games, rating drift during the day) improve playoff log loss by a further 1.8 (2022-24) and
2.4 (2025-26) ×1000, about 6 SE: real, but small next to M9's 9-10. Not shipped.

## 3. Live mode on the website: the big gap, now closed

Live mode fixed the games already played but simulated the rest with the pre-event strengths,
so the day's form was ignored. The engine now conditions every strength on the played games
(`web/src/engine/dayForm.ts`): before the event each strength is Normal(rating, sd² + form sd²);
each played game says clip(goal difference, ±6) = strength gap + pair shift + noise. The posterior
is Gaussian and is sampled once per run (Cholesky). With no games played nothing changes. The only
setting is the per-game noise sd: 3.4 goals, chosen on 2022-24 from 2, 2.6, 3.4, 4.5 and 6
(the rating's own value is 2.6; 2.6 and 3.4 are close).

Backtest (`m10_live.py`): every group-then-playoff event replayed with the shipped M9 setup,
forecasting the final placings from two points in the day, update off vs on (paired per event):

| window | point | events | winner log loss off → on | p | RPS p | top-3 Brier p | actual winner's chance off → on |
|---|---|---|---|---|---|---|---|
| 2022-24 | half the group games played | 660 | 1.0099 → 0.9745 | 0.0002 | <0.001 | 0.0002 | 36.4% → 37.7% |
| 2022-24 | playoff start | 660 | 1.0387 → 1.0116 | 0.006 | 0.005 | 0.07 | 35.4% → 36.4% |
| **2025-26** | half the group games played | 393 | 1.1699 → 1.1089 | 0.0003 | 0.003 | 0.05 | 31.0% → 33.0% |
| **2025-26** | playoff start | 393 | 1.2410 → 1.1315 | 0.007 | <0.001 | <0.001 | 28.9% → 32.3% |

Restricted to events where the default group rules reproduce the real playoff field, the gains
are the same size (2025-26 playoff start: 0.9545 → 0.8823, p = 0.047; RPS p < 0.001). Chances
are geometric means. The playoff-start numbers include events where the engine's default
advancement rules pick a different playoff field than the real one, which is why the "all"
column can look worse than mid-groups.
