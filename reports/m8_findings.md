# M8: what the h2h.puck.no ratings add

https://h2h.puck.no/ratings rates the same players (Scorpion, bordshockey and table-hockey.eu
games) with three ideas our model did not use:

1. **Glicko-2 game by game on a soft margin score**: s = 1 / (1 + e^(−0.4 × goal difference)),
   so a 1-goal win counts 0.60 and a 5-goal win 0.88. RD grows one volatility step per week
   of inactivity.
2. **Its forecast shrinks the gap by Glicko's g(RD)** before an ordered logit.
3. **A head-to-head edge per pair**: sum(result − expected) / (meetings + 20), added to the
   gap. Some players match up badly against one opponent, beyond what both ratings say.

The site has no download or API, and this environment can't reach it. So we rebuilt the
method from its published description and ran it on our own games. Every setting was fitted
on 2015–2021, compared on 2022–2024, and scored on 2025–2026 only at the end. This is the
same protocol as M2 and M7.

## Game level (3-way regulation log loss, Δ ×1000 vs the current model, negative = better)

| model | 2022–24 Δ (±SE) | 2025–26 Δ (±SE) |
|---|---|---|
| h2h.puck.no rating + forecast, published settings | −3.40 (0.25) | −3.42 (0.28) |
| same, settings tuned on our 2015–2021 games | −4.89 (0.40) | −5.84 (0.50) |
| head-to-head edge alone, on top of today's Kalman | −1.60 (0.11) | −1.95 (0.16) |
| (M7's best idea, results-only blend H11, for reference) | −1.41 (0.13) | −2.05 (0.14) |
| **composite rating**: 70% tuned Glicko-MoV + 30% Kalman, sd ×0.65 | −5.09 (0.33) | −5.75 (0.37) |
| **composite + head-to-head edge (shipped)** | **−6.10** | **−7.07** |

For scale, switching from Elo to Kalman was worth about −13. The shipped model gains about
half that again, which is four times the best M7 hypothesis. It also improves on every slice
we checked: pairs that had met 5+ times (−8 to −9), playoff games (−6 to −7), and both windows.
Full table and fitted settings are in `m8_puck.md`.

What tuning changed versus the published settings: a steeper margin curve (0.8 instead of 0.4),
faster volatility (τ 3.75), RD steps every 3.4 days, and a wider newcomer RD. A blend still
beats either rating on its own: the Kalman filter keeps a little information that Glicko-MoV
misses. Glicko's RD overstates the real uncertainty, so the blended sd is scaled by 0.65 (fitted
on 2015–2021). Without that scaling, integrating the uncertainty costs about 2 points.

## Tournament level (same 879 + 623 events and engine as M5/M7, 2,000 runs, form shock 0.45)

| variant | 2022–24 winner LL | RPS | top-3 Brier | 2025–26 winner LL | RPS | top-3 Brier |
|---|---|---|---|---|---|---|
| before: Kalman | **1.2076** | 0.0903 | 0.1074 | 1.2908 | 0.0956 | 0.1112 |
| composite | 1.2277 | 0.0884 | 0.1048 | 1.3212 | 0.0936 | 0.1096 |
| **after: composite + head-to-head** | 1.2163 | **0.0880** | **0.1045** | **1.2811** | **0.0936** | **0.1095** |

Paired per event, the shipped model against the old one:

- **RPS over finishing places**: −0.0023 and −0.0020, p < 0.001 in both windows. It is
  better in 61% and 63% of events.
- **Top-3 Brier**: −0.0028 (p < 0.001) and −0.0017 (p = 0.029).
- **Winner log loss**: +0.009 (p = 0.62) and −0.010 (p = 0.67). This is noise in both
  directions.

This is the first rating change that clearly moves the tournament backtest. M7's game-level
gains (about −1.5) disappeared at tournament level. A gain four times larger shows up in the
whole finishing order (RPS) and in who makes the podium. It does not show up in the single
winner metric, which a few upsets dominate. Head-to-head edges matter at tournament level.
Without them, the composite is worse on winner log loss. With them, it is the best variant on
2025–26.

Head-to-head edges in this backtest use the same filters as the website export (2+ earlier
meetings, |edge| ≥ 0.01 goals), so it measures what ships.

The form shock was rechecked for the new rating on 2022–24, with candidates 0.2, 0.3, 0.45, 0.6
and 0.75. 0.45 stays best.

## What changed in the product

- `thsim/ratings/composite.py`: the blend, the head-to-head edge, current ratings and the
  newcomer prior. Settings are in `reports/m8_params.json`, and `thsim.export` uses them when
  the file exists.
- `h2h.json` holds edges among exported players with 2+ meetings (about 71k pairs, 1.3 MB,
  edges frozen at export time). The website passes the edges among the chosen entrants to the
  engine. The engine adds the pair's edge to the strength gap in every simulated game between
  them (`TournamentConfig.h2h`).
- Displayed ratings change scale a little. The blend spreads the top players further apart:
  the best is now about 2200 instead of about 2000. Newcomers start nearer the average with a
  wider ±, as in Glicko.
- The About page names the method and shows the before/after backtest row.

## Not done

- **Their extra data source** (table-hockey.eu / resultat.bordshockey.se). It is likely worth
  checking for games our two scrapers miss, but it is out of reach from this environment.
- **Tuning the Glicko-MoV settings on the g(RD) forecast** instead of the integrated one. Their
  own forecast form scored better on the rating alone, so a retune there might gain a bit more.
