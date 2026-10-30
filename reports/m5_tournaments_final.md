# Milestone 5: tournament-level backtest (final)

Events reconstructed: {'seen': 1468, 'unsupported': 849, 'kept': 619, 'rule_ok': 571}. Every event is simulated with the website's engine (Node), ratings
from games before each player's first game of the event. `rule_ok` = the default advancement
rules reproduce the real playoff field from the real group results.
Best variant by winner log loss: `kalman_form0.45`.

## All reconstructed events

| variant | events | winner_ll | uniform_ll | rps | top3_brier | uniform_top3_brier | fav_won | mean_p_winner |
|---|---|---|---|---|---|---|---|---|
| elo | 619 | 1.4466 | 2.3444 | 0.0991 | 0.1133 | 0.2035 | 0.5347 | 0.4735 |
| kalman | 619 | 1.3177 | 2.3444 | 0.0962 | 0.1118 | 0.2035 | 0.5363 | 0.4549 |
| kalman_form0.45 | 619 | 1.2924 | 2.3444 | 0.0956 | 0.1113 | 0.2035 | 0.5396 | 0.4331 |

`winner_ll`: mean -log P(actual winner wins). `rps`: ranked probability score over the known places
(1, 2, 3, 4 and "lower"; whole table for leagues). `top3_brier`: Brier score of P(top 3). Uniform =
every player equally likely.

### By event kind

| variant | kind | winner_ll | uniform_ll | rps |
|---|---|---|---|---|
| elo | groups_ko | 1.5157 | 2.3392 | 0.1002 |
| elo | ko | 5.2862 | 2.7358 | 0.0775 |
| elo | league | 1.2562 | 2.3467 | 0.0976 |
| kalman | groups_ko | 1.3307 | 2.3392 | 0.0970 |
| kalman | ko | 1.9624 | 2.7358 | 0.0600 |
| kalman | league | 1.2835 | 2.3467 | 0.0955 |
| kalman_form0.45 | groups_ko | 1.3031 | 2.3392 | 0.0964 |
| kalman_form0.45 | ko | 2.0235 | 2.7358 | 0.0618 |
| kalman_form0.45 | league | 1.2606 | 2.3467 | 0.0948 |

### Events where the rules reproduce who advanced

| variant | events | winner_ll | uniform_ll | rps | top3_brier | uniform_top3_brier | fav_won | mean_p_winner |
|---|---|---|---|---|---|---|---|---|
| elo | 571 | 1.2277 | 2.3388 | 0.0878 | 0.0990 | 0.2043 | 0.5569 | 0.4890 |
| kalman | 571 | 1.2036 | 2.3388 | 0.0862 | 0.0990 | 0.2043 | 0.5552 | 0.4710 |
| kalman_form0.45 | 571 | 1.1783 | 2.3388 | 0.0859 | 0.0989 | 0.2043 | 0.5569 | 0.4479 |

## Paired per-event test on winner log loss (negative = first variant better)

| comparison | events | mean diff | t-test p | Wilcoxon p |
|---|---|---|---|---|
| elo vs kalman | 619 | +0.1288 | 0.00592 | 0.63 |
| elo vs kalman_form0.45 | 619 | +0.1542 | 0.00054 | 0.104 |
| kalman vs kalman_form0.45 | 619 | +0.0253 | 0.147 | 2.54e-05 |

## Calibration (kalman_form0.45)

P(win):

| forecast | observed | player-events |
|---|---|---|
| 0.014 | 0.017 | 5,765 |
| 0.142 | 0.141 | 512 |
| 0.244 | 0.303 | 221 |
| 0.347 | 0.322 | 174 |
| 0.447 | 0.404 | 156 |
| 0.551 | 0.546 | 108 |
| 0.649 | 0.612 | 98 |
| 0.748 | 0.682 | 110 |
| 0.845 | 0.754 | 61 |
| 0.935 | 0.852 | 27 |

P(top 3):

| forecast | observed | player-events |
|---|---|---|
| 0.022 | 0.027 | 3,609 |
| 0.145 | 0.150 | 672 |
| 0.247 | 0.225 | 570 |
| 0.346 | 0.372 | 400 |
| 0.449 | 0.476 | 313 |
| 0.549 | 0.560 | 259 |
| 0.648 | 0.636 | 261 |
| 0.748 | 0.780 | 264 |
| 0.852 | 0.806 | 283 |
| 0.961 | 0.918 | 601 |
