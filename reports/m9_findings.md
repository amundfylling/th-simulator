# M9: a LightGBM layer on top of the ratings

A prediction experiment in another project (a LightGBM model over Glicko/Elo ratings and about
60 pre-game features, scored on the scraped matches) beat a Glicko-2 + head-to-head forecast by
5% in log loss. This milestone reruns those ideas on our own data, with our protocol:

- everything is fitted on 2015–2021 (LightGBM: trained on 2015–2020, early-stopped on 2021);
- choices are made on 2022–2024;
- 2025–2026 is the untouched holdout.

The baseline is the model that shipped in M8: the composite rating plus the head-to-head edge.

## Two warnings from the experiment, checked first

- **Playoff draws.** Our model never gave playoff games a flat 12% draw chance. Playoff
  games have their own regulation model. A regulation tie goes to golden-goal overtime, which
  13.5% of our playoff games reach, and the engine plays that overtime. Only 0.6% of playoff
  games end drawn.
- **Player order leaks the result.** In Trefik and galdahokejs, player 1 wins about 70% of
  round-robin games, so the order gives the result away. Neither source is in our data. In
  our round-robin games player 1 wins 44% and player 2 wins 43%. In playoffs player 1 wins
  about 60%, because player 1 is usually the higher seed. The rating model only uses the
  rating gap, symmetrically. LightGBM cannot use the order either: every training game appears
  from both sides, and every prediction averages both views.

## Game level (3-way regulation log loss, Δ ×1000 vs M8, negative = better)

| model | 2022–24 Δ (±SE) | 2025–26 Δ (±SE) | newcomer games | favourite won, 2025–26 |
|---|---|---|---|---|
| per-player draw rate (moves the draw cut) | −0.46 (0.08) | −0.48 (0.12) | −2 to −3 | = |
| extra uncertainty for players with few games | −0.01 | −0.02 | ~0 | = |
| recent form (residual EWMA over 30 games) | −0.62 (0.09) | −0.78 (0.10) | ~0 | +0.06 pt |
| all three as simple terms | −1.07 (0.13) | −1.27 (0.15) | −2 to −3 | +0.05 pt |
| **LightGBM layer** | **−10.23 (0.59)** | **−8.97 (0.65)** | **−37 / −38** | **75.84% → 76.12%** |

LightGBM is worth about as much as the M8 update again, and almost as much as the original
switch from Elo to the Kalman rating (−13). Most of its gain comes from games where one
player has fewer than 30 earlier games: −37 there, against about −6 elsewhere. Its inputs, in
order of importance, are:

- the composite gap (72% of the gain);
- the separate Kalman and Elo ratings (13% between them), which carry information about
  newcomers that the blend loses;
- in-event and recent form;
- games played, Glicko RD, age, draw rate and the head-to-head record.

Per-player draw rate and recent form are real but small as standalone terms, and the model
already uses both.

Details: `m9_features.md`.

## Tournament level (same events, engine, game model and form shock as M5/M8)

For a pre-tournament forecast the model is retrained on **frozen features**. Head-to-head,
form and games played are taken as they stood before the event's first game, and in-event
form is left out. For every pair of entrants, the model's win/draw/loss odds become a shift on
top of the rating gap. The shift goes through the engine's head-to-head map, so it applies in
every simulated game between those two players. The backtest calls the TypeScript code the
website runs (`web/src/engine/gbm.ts`). That code reproduces LightGBM's own predictions to
within 4·10⁻⁶.

| variant | 2022–24 winner LL | RPS | top-3 Brier | 2025–26 winner LL | RPS | top-3 Brier |
|---|---|---|---|---|---|---|
| goal-difference rating (M5) | 1.2076 | 0.0903 | 0.1074 | 1.2908 | 0.0956 | 0.1112 |
| composite + head-to-head (M8) | 1.2163 | 0.0880 | 0.1045 | 1.2811 | 0.0936 | 0.1095 |
| **+ LightGBM layer** | **1.1451** | **0.0853** | **0.1004** | **1.2067** | **0.0908** | **0.1055** |

Compared with M8, paired per event:

- **Winner log loss:** −0.071 (p < 0.001) on 2022–24 and −0.074 (p = 0.003) on 2025–26.
- **RPS and top-3 Brier:** p < 0.001 on both windows. RPS is better in 59% of events on
  both windows; top-3 Brier in 60% and 57%.

In plain terms, the actual winner had been given, on average:

| model | 2022–24 | 2025–26 |
|---|---|---|
| M5 | 29.9% | 27.5% |
| M8 | 29.6% | 27.8% |
| M9 (with the LightGBM layer) | **31.8%** | **29.9%** |

These are geometric means. This is the first change to clearly move the winner metric.

## What ships

- `thsim/models/gbm.py` trains the layer and exports it as compact trees (`gbm.json`, 2.3 MB,
  about 0.8 MB gzipped).
  - It also exports each player's current feature vector. The vector is computed by "probe"
    walkovers on the export date, so it uses exactly the code that built the training features.
  - It also exports a newcomer vector for players imported without history.
  - The nightly export retrains the model on the last 10 years for the stored number of
    rounds (284) in about a minute. `lightgbm` is now a dependency.
- `web/src/engine/gbm.ts` evaluates the trees and computes pair shifts in the simulation
  worker. Odds for 64 players and 10,000 runs still take about 4 s in a headless browser.
- `h2h.json` entries are now `[edge, meetings]`. The website still reads the old number format.
- The About page explains the layer, and the backtest table gains a "ratings + head-to-head
  only" row.

## Not done

- **Playoff-series state.** This only matters live, inside a series, and the engine doesn't
  model it per game.
- **The extra ~30 features from the other project.** Each one we tested as a simple term was
  small. A larger feature set or more tuning may still gain something, but should be checked
  the same way first.
