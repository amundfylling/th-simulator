# Milestone 5: tournament-level backtest (select)

Events reconstructed: {'seen': 2041, 'unsupported': 1162, 'kept': 879, 'rule_ok': 813}. Every event is simulated with the website's engine (Node), ratings
from games before each player's first game of the event. `rule_ok` = the default advancement
rules reproduce the real playoff field from the real group results.
Best variant by winner log loss: `kalman_form0.45`.

## All reconstructed events

| variant | events | winner_ll | uniform_ll | rps | top3_brier | uniform_top3_brier | fav_won | mean_p_winner |
|---|---|---|---|---|---|---|---|---|
| elo | 879 | 1.3487 | 2.3565 | 0.0935 | 0.1107 | 0.2027 | 0.5472 | 0.4687 |
| kalman | 879 | 1.2327 | 2.3565 | 0.0905 | 0.1078 | 0.2027 | 0.5381 | 0.4498 |
| kalman_form0.15 | 879 | 1.2174 | 2.3565 | 0.0904 | 0.1077 | 0.2027 | 0.5484 | 0.4477 |
| kalman_form0.3 | 879 | 1.2159 | 2.3565 | 0.0903 | 0.1075 | 0.2027 | 0.5461 | 0.4396 |
| kalman_form0.45 | 879 | 1.2076 | 2.3565 | 0.0903 | 0.1074 | 0.2027 | 0.5529 | 0.4291 |
| kalman_form0.6 | 879 | 1.2099 | 2.3565 | 0.0905 | 0.1075 | 0.2027 | 0.5415 | 0.4148 |
| kalman_nounc | 879 | 1.2731 | 2.3565 | 0.0915 | 0.1089 | 0.2027 | 0.5495 | 0.4630 |

`winner_ll`: mean -log P(actual winner wins). `rps`: ranked probability score over the known places
(1, 2, 3, 4 and "lower"; whole table for leagues). `top3_brier`: Brier score of P(top 3). Uniform =
every player equally likely.

### By event kind

| variant | kind | winner_ll | uniform_ll | rps |
|---|---|---|---|---|
| elo | groups_ko | 1.1934 | 2.3148 | 0.0909 |
| elo | ko | 3.4323 | 2.8426 | 0.1352 |
| elo | league | 1.7791 | 2.4738 | 0.1007 |
| kalman | groups_ko | 1.1102 | 2.3148 | 0.0887 |
| kalman | ko | 3.1937 | 2.8426 | 0.1303 |
| kalman | league | 1.5646 | 2.4738 | 0.0952 |
| kalman_form0.15 | groups_ko | 1.0927 | 2.3148 | 0.0886 |
| kalman_form0.15 | ko | 3.1975 | 2.8426 | 0.1294 |
| kalman_form0.15 | league | 1.5556 | 2.4738 | 0.0953 |
| kalman_form0.3 | groups_ko | 1.0965 | 2.3148 | 0.0885 |
| kalman_form0.3 | ko | 3.0145 | 2.8426 | 0.1266 |
| kalman_form0.3 | league | 1.5419 | 2.4738 | 0.0951 |
| kalman_form0.45 | groups_ko | 1.0904 | 2.3148 | 0.0885 |
| kalman_form0.45 | ko | 2.9021 | 2.8426 | 0.1253 |
| kalman_form0.45 | league | 1.5296 | 2.4738 | 0.0950 |
| kalman_form0.6 | groups_ko | 1.0965 | 2.3148 | 0.0887 |
| kalman_form0.6 | ko | 2.7837 | 2.8426 | 0.1218 |
| kalman_form0.6 | league | 1.5228 | 2.4738 | 0.0952 |
| kalman_nounc | groups_ko | 1.1232 | 2.3148 | 0.0895 |
| kalman_nounc | ko | 3.4784 | 2.8426 | 0.1339 |
| kalman_nounc | league | 1.6841 | 2.4738 | 0.0967 |

### Events where the rules reproduce who advanced

| variant | events | winner_ll | uniform_ll | rps | top3_brier | uniform_top3_brier | fav_won | mean_p_winner |
|---|---|---|---|---|---|---|---|---|
| elo | 813 | 1.2571 | 2.3476 | 0.0892 | 0.1054 | 0.2037 | 0.5560 | 0.4769 |
| kalman | 813 | 1.1595 | 2.3476 | 0.0864 | 0.1028 | 0.2037 | 0.5474 | 0.4576 |
| kalman_form0.15 | 813 | 1.1551 | 2.3476 | 0.0864 | 0.1027 | 0.2037 | 0.5597 | 0.4553 |
| kalman_form0.3 | 813 | 1.1525 | 2.3476 | 0.0863 | 0.1026 | 0.2037 | 0.5572 | 0.4471 |
| kalman_form0.45 | 813 | 1.1485 | 2.3476 | 0.0864 | 0.1026 | 0.2037 | 0.5633 | 0.4363 |
| kalman_form0.6 | 813 | 1.1539 | 2.3476 | 0.0867 | 0.1030 | 0.2037 | 0.5510 | 0.4218 |
| kalman_nounc | 813 | 1.2057 | 2.3476 | 0.0872 | 0.1037 | 0.2037 | 0.5597 | 0.4705 |

## Paired per-event test on winner log loss (negative = first variant better)

| comparison | events | mean diff | t-test p | Wilcoxon p |
|---|---|---|---|---|
| elo vs kalman | 879 | +0.1160 | 7.81e-06 | 0.363 |
| elo vs kalman_form0.15 | 879 | +0.1313 | 3.59e-06 | 0.313 |
| elo vs kalman_form0.3 | 879 | +0.1328 | 3.98e-06 | 0.945 |
| elo vs kalman_form0.45 | 879 | +0.1411 | 3.31e-06 | 0.506 |
| elo vs kalman_form0.6 | 879 | +0.1389 | 1.24e-05 | 0.105 |
| elo vs kalman_nounc | 879 | +0.0756 | 0.0018 | 0.195 |
| kalman vs kalman_form0.15 | 879 | +0.0154 | 0.112 | 0.25 |
| kalman vs kalman_form0.3 | 879 | +0.0168 | 0.062 | 3.34e-09 |
| kalman vs kalman_form0.45 | 879 | +0.0251 | 0.0291 | 4.64e-10 |
| kalman vs kalman_form0.6 | 879 | +0.0229 | 0.0835 | 8.52e-13 |
| kalman vs kalman_nounc | 879 | -0.0404 | 0.0126 | 0.00035 |
| kalman_form0.15 vs kalman_form0.3 | 879 | +0.0015 | 0.671 | 8.69e-12 |
| kalman_form0.15 vs kalman_form0.45 | 879 | +0.0097 | 0.0776 | 2.09e-12 |
| kalman_form0.15 vs kalman_form0.6 | 879 | +0.0075 | 0.353 | 2.81e-15 |
| kalman_form0.15 vs kalman_nounc | 879 | -0.0558 | 0.000299 | 0.000562 |
| kalman_form0.3 vs kalman_form0.45 | 879 | +0.0082 | 0.0477 | 4.44e-09 |
| kalman_form0.3 vs kalman_form0.6 | 879 | +0.0060 | 0.353 | 1.14e-13 |
| kalman_form0.3 vs kalman_nounc | 879 | -0.0573 | 0.000441 | 3.06e-06 |
| kalman_form0.45 vs kalman_form0.6 | 879 | -0.0022 | 0.546 | 4.08e-16 |
| kalman_form0.45 vs kalman_nounc | 879 | -0.0655 | 0.000382 | 2.96e-07 |
| kalman_form0.6 vs kalman_nounc | 879 | -0.0633 | 0.00183 | 1.54e-09 |

## Calibration (kalman_form0.45)

P(win):

| forecast | observed | player-events |
|---|---|---|
| 0.013 | 0.014 | 8,075 |
| 0.144 | 0.142 | 811 |
| 0.244 | 0.269 | 435 |
| 0.346 | 0.311 | 312 |
| 0.449 | 0.454 | 227 |
| 0.548 | 0.503 | 157 |
| 0.649 | 0.588 | 114 |
| 0.747 | 0.765 | 98 |
| 0.850 | 0.926 | 68 |
| 0.942 | 0.945 | 55 |

P(top 3):

| forecast | observed | player-events |
|---|---|---|
| 0.021 | 0.025 | 5,169 |
| 0.145 | 0.144 | 952 |
| 0.248 | 0.238 | 631 |
| 0.350 | 0.358 | 578 |
| 0.450 | 0.429 | 487 |
| 0.549 | 0.588 | 422 |
| 0.647 | 0.667 | 430 |
| 0.750 | 0.721 | 402 |
| 0.850 | 0.838 | 462 |
| 0.959 | 0.938 | 819 |
