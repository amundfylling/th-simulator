# Milestone 1: data facts

Games after cleaning and cross-source dedupe: 1,287,788 (scorpion 979,778, bordshockey 308,010), of which 1,410 walkovers (kept for structure, excluded from fitting).
Gap = pre-game Elo difference (K=24, K=64 decaying over a player's first 40 games). "Seasoned" = both players have 30+ prior games.

| Fact | Expected | Measured |
|---|---|---|
| Round-robin draw rate | 12.5% | 12.4% |
| RR draws, equal players (gap < 50) | ~18% | 18.1% |
| RR draws, gap 600+ | < 2% | 1.9% |
| Playoff games to golden-goal OT | 13.2% | 13.5% |
| OT games won by exactly one goal | 100% | 99.98% |
| Playoff games drawn without OT | ~1% | 0.62% |
| Favourite wins, gap 200-300, regulation playoff | 83% | 82.2% |
| Favourite wins, gap 200-300, overtime | 67% | 66.4% |
| Goals per player: mean | 2.77 | 2.78 |
| Goals per player: variance | 4.26 | 4.46 |
| Score correlation, equal players | -0.10 | -0.11 |
| Total goals, gap 0-50 to 600+ | 5.1 to 6.9 | 5.2 to 7.0 |
| RR groups complete | 89% | 90.0% |
| Median group size | 9 | 9 |
| Multi-stage tournaments | most | 68.0% |
| Max stages in a tournament | 10 | 17 |
| Baseline Elo + ordered logit, log loss (3-way) | 0.791 | 0.783 |
| Base rates, log loss | 0.995 | 0.992 |
| Baseline RPS | 0.170 | 0.169 |

Baseline fit on 405,956 seasoned round-robin games 2015-2021 (beta=0.00582, c=0.367), evaluated on 302,602 games 2022-2026. Brier 0.449, base-rate RPS 0.246.

## Round-robin draws by gap (seasoned players)

| gap_bin | mean | size |
|---|---|---|
| 0-50 | 0.181 | 130289.000 |
| 50-100 | 0.173 | 123777.000 |
| 100-200 | 0.153 | 208392.000 |
| 200-300 | 0.118 | 146835.000 |
| 300-400 | 0.082 | 96372.000 |
| 400-600 | 0.047 | 93454.000 |
| 600+ | 0.019 | 34528.000 |

## Favourite win share in decided playoff games

| gap_bin | regulation | overtime |
|---|---|---|
| 0-50 | 0.541 | 0.526 |
| 50-100 | 0.621 | 0.565 |
| 100-200 | 0.717 | 0.599 |
| 200-300 | 0.822 | 0.664 |
| 300-400 | 0.891 | 0.710 |
| 400-600 | 0.944 | 0.776 |
| 600+ | 0.981 | 0.795 |

## Total goals by gap (round-robin, seasoned)

| gap_bin | total |
|---|---|
| 0-50 | 5.156 |
| 50-100 | 5.195 |
| 100-200 | 5.343 |
| 200-300 | 5.625 |
| 300-400 | 5.962 |
| 400-600 | 6.406 |
| 600+ | 7.031 |

## Playoff series length (wins needed per round)

| wins_needed | count |
|---|---|
| 1 | 2908 |
| 2 | 11347 |
| 3 | 14379 |
| 4 | 9650 |
| 5 | 35 |
| 6 | 2 |
| 9 | 1 |
| 10 | 4 |
| 15 | 1 |

86,693 decided series. 99.0% of rounds have every series winner on the same win count (the rest mix formats or include abandoned series).

## Round-robin groups

15,070 groups (stages split by connected components). 90.0% complete, 22.5% complete double round-robins, median size 9.

## Stages per tournament

| stage_seq | count |
|---|---|
| 1 | 3053 |
| 2 | 3473 |
| 3 | 1452 |
| 4 | 722 |
| 5 | 435 |
| 6 | 195 |
| 7 | 118 |
| 8 | 35 |
| 9 | 26 |
| 10 | 12 |
| 11 | 4 |
| 12 | 2 |
| 13 | 1 |
| 14 | 3 |
| 15 | 3 |
| 17 | 1 |
