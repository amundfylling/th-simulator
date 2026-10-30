# Milestone 5: tournament-level backtest (final)

Events reconstructed: {'seen': 1468, 'unsupported': 849, 'kept': 619, 'rule_ok': 571}. Every event is simulated with the website's engine (Node), ratings
from games before each player's first game of the event. `rule_ok` = the default advancement
rules reproduce the real playoff field from the real group results.
Best variant by winner log loss: `kalman_form0.45`.

## All reconstructed events

| variant | events | winner_ll | uniform_ll | rps | top3_brier | uniform_top3_brier | fav_won | mean_p_winner |
|---|---|---|---|---|---|---|---|---|
| elo | 619 | 1.4797 | 2.3444 | 0.0991 | 0.1163 | 0.2050 | 0.5347 | 0.4735 |
| kalman | 619 | 1.3156 | 2.3444 | 0.0962 | 0.1144 | 0.2050 | 0.5380 | 0.4545 |
| kalman_form0.45 | 619 | 1.2959 | 2.3444 | 0.0956 | 0.1138 | 0.2050 | 0.5380 | 0.4325 |

`winner_ll`: mean -log P(actual winner wins). `rps`: ranked probability score over the known places
(1, 2, 3, 4 and "lower"; whole table for leagues). `top3_brier`: Brier score of P(top 3). Uniform =
every player equally likely.

### By event kind

| variant | kind | winner_ll | uniform_ll | rps |
|---|---|---|---|---|
| elo | groups_ko | 1.5656 | 2.3392 | 0.1002 |
| elo | ko | 5.2852 | 2.7358 | 0.0778 |
| elo | league | 1.2603 | 2.3467 | 0.0976 |
| kalman | groups_ko | 1.3273 | 2.3392 | 0.0969 |
| kalman | ko | 1.9641 | 2.7358 | 0.0607 |
| kalman | league | 1.2835 | 2.3467 | 0.0955 |
| kalman_form0.45 | groups_ko | 1.3085 | 2.3392 | 0.0965 |
| kalman_form0.45 | ko | 2.0306 | 2.7358 | 0.0620 |
| kalman_form0.45 | league | 1.2606 | 2.3467 | 0.0948 |

### Events where the rules reproduce who advanced

| variant | events | winner_ll | uniform_ll | rps | top3_brier | uniform_top3_brier | fav_won | mean_p_winner |
|---|---|---|---|---|---|---|---|---|
| elo | 571 | 1.2528 | 2.3388 | 0.0878 | 0.1018 | 0.2058 | 0.5569 | 0.4891 |
| kalman | 571 | 1.2010 | 2.3388 | 0.0862 | 0.1014 | 0.2058 | 0.5552 | 0.4705 |
| kalman_form0.45 | 571 | 1.1800 | 2.3388 | 0.0859 | 0.1013 | 0.2058 | 0.5569 | 0.4473 |

## Paired per-event test on winner log loss (negative = first variant better)

| comparison | events | mean diff | t-test p | Wilcoxon p |
|---|---|---|---|---|
| elo vs kalman | 619 | +0.1640 | 0.00149 | 0.873 |
| elo vs kalman_form0.45 | 619 | +0.1838 | 0.000236 | 0.132 |
| kalman vs kalman_form0.45 | 619 | +0.0197 | 0.246 | 2.46e-05 |

## Calibration (kalman_form0.45)

P(win):

| forecast | observed | player-events |
|---|---|---|
| 0.014 | 0.017 | 5,764 |
| 0.141 | 0.141 | 509 |
| 0.243 | 0.298 | 225 |
| 0.346 | 0.325 | 169 |
| 0.446 | 0.405 | 163 |
| 0.551 | 0.556 | 108 |
| 0.651 | 0.589 | 95 |
| 0.747 | 0.694 | 111 |
| 0.844 | 0.738 | 61 |
| 0.934 | 0.852 | 27 |

P(top 3):

| forecast | observed | player-events |
|---|---|---|
| 0.021 | 0.030 | 3,683 |
| 0.144 | 0.169 | 698 |
| 0.247 | 0.251 | 550 |
| 0.344 | 0.379 | 388 |
| 0.449 | 0.477 | 310 |
| 0.549 | 0.604 | 250 |
| 0.648 | 0.648 | 267 |
| 0.747 | 0.798 | 257 |
| 0.852 | 0.826 | 270 |
| 0.960 | 0.911 | 559 |
