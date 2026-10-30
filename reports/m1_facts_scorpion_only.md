# Milestone 1: data facts

Games after cleaning and cross-source dedupe: 1,019,266 (scorpion 1,019,266), of which 1,628 walkovers (kept for structure, excluded from fitting).
Gap = pre-game Elo difference (K=24, K=64 decaying over a player's first 40 games). "Seasoned" = both players have 30+ prior games.

| Fact | Expected | Measured |
|---|---|---|
| Round-robin draw rate | 12.5% | 12.5% |
| RR draws, equal players (gap < 50) | ~18% | 18.0% |
| RR draws, gap 600+ | < 2% | 1.9% |
| Playoff games to golden-goal OT | 13.2% | 13.2% |
| OT games won by exactly one goal | 100% | 100.00% |
| Playoff games drawn without OT | ~1% | 0.90% |
| Favourite wins, gap 200-300, regulation playoff | 83% | 82.9% |
| Favourite wins, gap 200-300, overtime | 67% | 66.5% |
| Goals per player: mean | 2.77 | 2.74 |
| Goals per player: variance | 4.26 | 4.50 |
| Score correlation, equal players | -0.10 | -0.11 |
| Total goals, gap 0-50 to 600+ | 5.1 to 6.9 | 5.1 to 7.0 |
| RR groups complete | 89% | 89.3% |
| Median group size | 9 | 9 |
| Multi-stage tournaments | most | 74.7% |
| Max stages in a tournament | 10 | 17 |
| Baseline Elo + ordered logit, log loss (3-way) | 0.791 | 0.791 |
| Base rates, log loss | 0.995 | 0.995 |
| Baseline RPS | 0.170 | 0.170 |

Baseline fit on 348,270 seasoned round-robin games 2015-2021 (beta=0.00589, c=0.365), evaluated on 239,814 games 2022-2026. Brier 0.455, base-rate RPS 0.246.

## Round-robin draws by gap (seasoned players)

| gap_bin | mean | size |
|---|---|---|
| 0-50 | 0.180 | 106905.000 |
| 50-100 | 0.174 | 101594.000 |
| 100-200 | 0.153 | 169728.000 |
| 200-300 | 0.117 | 118438.000 |
| 300-400 | 0.080 | 76463.000 |
| 400-600 | 0.047 | 71782.000 |
| 600+ | 0.019 | 23801.000 |

## Favourite win share in decided playoff games

| gap_bin | regulation | overtime |
|---|---|---|
| 0-50 | 0.546 | 0.531 |
| 50-100 | 0.623 | 0.563 |
| 100-200 | 0.720 | 0.601 |
| 200-300 | 0.829 | 0.665 |
| 300-400 | 0.890 | 0.711 |
| 400-600 | 0.941 | 0.792 |
| 600+ | 0.979 | 0.846 |

## Total goals by gap (round-robin, seasoned)

| gap_bin | total |
|---|---|
| 0-50 | 5.113 |
| 50-100 | 5.161 |
| 100-200 | 5.305 |
| 200-300 | 5.599 |
| 300-400 | 5.940 |
| 400-600 | 6.393 |
| 600+ | 6.986 |

## Playoff series length (wins needed per round)

| wins_needed | count |
|---|---|
| 1 | 2350 |
| 2 | 9281 |
| 3 | 7610 |
| 4 | 6665 |
| 5 | 16 |
| 6 | 2 |
| 9 | 1 |
| 10 | 4 |
| 15 | 1 |

62,807 decided series. 98.8% of rounds have every series winner on the same win count (the rest mix formats or include abandoned series).

## Round-robin groups

12,713 groups (stages split by connected components). 89.3% complete, 20.0% complete double round-robins, median size 9.

## Stages per tournament

| stage_seq | count |
|---|---|
| 1 | 1651 |
| 2 | 2089 |
| 3 | 1296 |
| 4 | 675 |
| 5 | 423 |
| 6 | 194 |
| 7 | 120 |
| 8 | 35 |
| 9 | 27 |
| 10 | 12 |
| 11 | 4 |
| 12 | 2 |
| 13 | 1 |
| 14 | 3 |
| 15 | 3 |
| 17 | 1 |
