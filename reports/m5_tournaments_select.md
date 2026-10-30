# Milestone 5: tournament-level backtest (select)

Events reconstructed: {'seen': 2041, 'unsupported': 1162, 'kept': 879, 'rule_ok': 813}. Every event is simulated with the website's engine (Node), ratings
from games before each player's first game of the event. `rule_ok` = the default advancement
rules reproduce the real playoff field from the real group results.
Best variant by winner log loss: `kalman_form0.45`.

## All reconstructed events

| variant | events | winner_ll | uniform_ll | rps | top3_brier | uniform_top3_brier | fav_won | mean_p_winner |
|---|---|---|---|---|---|---|---|---|
| elo | 879 | 1.3506 | 2.3565 | 0.0935 | 0.1167 | 0.2055 | 0.5506 | 0.4693 |
| kalman | 879 | 1.2204 | 2.3565 | 0.0906 | 0.1130 | 0.2055 | 0.5392 | 0.4493 |
| kalman_form0.15 | 879 | 1.2188 | 2.3565 | 0.0905 | 0.1129 | 0.2055 | 0.5438 | 0.4474 |
| kalman_form0.3 | 879 | 1.2141 | 2.3565 | 0.0903 | 0.1127 | 0.2055 | 0.5484 | 0.4396 |
| kalman_form0.45 | 879 | 1.2059 | 2.3565 | 0.0903 | 0.1123 | 0.2055 | 0.5472 | 0.4293 |
| kalman_form0.6 | 879 | 1.2098 | 2.3565 | 0.0904 | 0.1122 | 0.2055 | 0.5438 | 0.4145 |
| kalman_nounc | 879 | 1.2734 | 2.3565 | 0.0915 | 0.1143 | 0.2055 | 0.5472 | 0.4629 |

`winner_ll`: mean -log P(actual winner wins). `rps`: ranked probability score over the known places
(1, 2, 3, 4 and "lower"; whole table for leagues). `top3_brier`: Brier score of P(top 3). Uniform =
every player equally likely.

### By event kind

| variant | kind | winner_ll | uniform_ll | rps |
|---|---|---|---|---|
| elo | groups_ko | 1.1944 | 2.3148 | 0.0908 |
| elo | ko | 3.4284 | 2.8426 | 0.1354 |
| elo | league | 1.7837 | 2.4738 | 0.1007 |
| kalman | groups_ko | 1.0938 | 2.3148 | 0.0888 |
| kalman | ko | 3.1947 | 2.8426 | 0.1301 |
| kalman | league | 1.5647 | 2.4738 | 0.0952 |
| kalman_form0.15 | groups_ko | 1.0945 | 2.3148 | 0.0887 |
| kalman_form0.15 | ko | 3.2101 | 2.8426 | 0.1295 |
| kalman_form0.15 | league | 1.5556 | 2.4738 | 0.0953 |
| kalman_form0.3 | groups_ko | 1.0943 | 2.3148 | 0.0885 |
| kalman_form0.3 | ko | 3.0012 | 2.8426 | 0.1265 |
| kalman_form0.3 | league | 1.5419 | 2.4738 | 0.0951 |
| kalman_form0.45 | groups_ko | 1.0882 | 2.3148 | 0.0884 |
| kalman_form0.45 | ko | 2.8894 | 2.8426 | 0.1263 |
| kalman_form0.45 | league | 1.5296 | 2.4738 | 0.0950 |
| kalman_form0.6 | groups_ko | 1.0965 | 2.3148 | 0.0887 |
| kalman_form0.6 | ko | 2.7640 | 2.8426 | 0.1215 |
| kalman_form0.6 | league | 1.5228 | 2.4738 | 0.0952 |
| kalman_nounc | groups_ko | 1.1236 | 2.3148 | 0.0894 |
| kalman_nounc | ko | 3.4676 | 2.8426 | 0.1339 |
| kalman_nounc | league | 1.6841 | 2.4738 | 0.0967 |

### Events where the rules reproduce who advanced

| variant | events | winner_ll | uniform_ll | rps | top3_brier | uniform_top3_brier | fav_won | mean_p_winner |
|---|---|---|---|---|---|---|---|---|
| elo | 813 | 1.2539 | 2.3476 | 0.0892 | 0.1112 | 0.2064 | 0.5572 | 0.4774 |
| kalman | 813 | 1.1628 | 2.3476 | 0.0865 | 0.1078 | 0.2064 | 0.5486 | 0.4569 |
| kalman_form0.15 | 813 | 1.1548 | 2.3476 | 0.0864 | 0.1078 | 0.2064 | 0.5535 | 0.4550 |
| kalman_form0.3 | 813 | 1.1521 | 2.3476 | 0.0863 | 0.1076 | 0.2064 | 0.5584 | 0.4470 |
| kalman_form0.45 | 813 | 1.1472 | 2.3476 | 0.0863 | 0.1073 | 0.2064 | 0.5572 | 0.4363 |
| kalman_form0.6 | 813 | 1.1530 | 2.3476 | 0.0867 | 0.1074 | 0.2064 | 0.5547 | 0.4215 |
| kalman_nounc | 813 | 1.2057 | 2.3476 | 0.0872 | 0.1089 | 0.2064 | 0.5572 | 0.4704 |

## Paired per-event test on winner log loss (negative = first variant better)

| comparison | events | mean diff | t-test p | Wilcoxon p |
|---|---|---|---|---|
| elo vs kalman | 879 | +0.1302 | 2.12e-06 | 0.3 |
| elo vs kalman_form0.15 | 879 | +0.1318 | 1.41e-06 | 0.325 |
| elo vs kalman_form0.3 | 879 | +0.1365 | 1.72e-06 | 0.945 |
| elo vs kalman_form0.45 | 879 | +0.1447 | 1.48e-06 | 0.603 |
| elo vs kalman_form0.6 | 879 | +0.1408 | 7.43e-06 | 0.13 |
| elo vs kalman_nounc | 879 | +0.0772 | 0.00123 | 0.141 |
| kalman vs kalman_form0.15 | 879 | +0.0016 | 0.838 | 0.56 |
| kalman vs kalman_form0.3 | 879 | +0.0063 | 0.459 | 1.22e-07 |
| kalman vs kalman_form0.45 | 879 | +0.0145 | 0.154 | 3.64e-09 |
| kalman vs kalman_form0.6 | 879 | +0.0107 | 0.368 | 1.5e-11 |
| kalman vs kalman_nounc | 879 | -0.0530 | 0.000918 | 0.000419 |
| kalman_form0.15 vs kalman_form0.3 | 879 | +0.0047 | 0.167 | 4.04e-10 |
| kalman_form0.15 vs kalman_form0.45 | 879 | +0.0129 | 0.0253 | 4.69e-12 |
| kalman_form0.15 vs kalman_form0.6 | 879 | +0.0090 | 0.283 | 2.2e-14 |
| kalman_form0.15 vs kalman_nounc | 879 | -0.0546 | 0.000283 | 0.00153 |
| kalman_form0.3 vs kalman_form0.45 | 879 | +0.0082 | 0.0404 | 5.48e-10 |
| kalman_form0.3 vs kalman_form0.6 | 879 | +0.0043 | 0.489 | 5.75e-14 |
| kalman_form0.3 vs kalman_nounc | 879 | -0.0593 | 0.000243 | 7.48e-06 |
| kalman_form0.45 vs kalman_form0.6 | 879 | -0.0038 | 0.301 | 8.7e-16 |
| kalman_form0.45 vs kalman_nounc | 879 | -0.0675 | 0.000235 | 1.12e-06 |
| kalman_form0.6 vs kalman_nounc | 879 | -0.0636 | 0.00169 | 6.56e-09 |

## Calibration (kalman_form0.45)

P(win):

| forecast | observed | player-events |
|---|---|---|
| 0.013 | 0.014 | 8,085 |
| 0.144 | 0.144 | 803 |
| 0.245 | 0.276 | 434 |
| 0.346 | 0.293 | 311 |
| 0.449 | 0.467 | 227 |
| 0.548 | 0.500 | 156 |
| 0.648 | 0.591 | 115 |
| 0.747 | 0.763 | 97 |
| 0.849 | 0.942 | 69 |
| 0.942 | 0.927 | 55 |

P(top 3):

| forecast | observed | player-events |
|---|---|---|
| 0.020 | 0.030 | 5,346 |
| 0.146 | 0.169 | 964 |
| 0.247 | 0.276 | 641 |
| 0.350 | 0.381 | 577 |
| 0.449 | 0.514 | 496 |
| 0.551 | 0.612 | 423 |
| 0.648 | 0.723 | 400 |
| 0.751 | 0.745 | 400 |
| 0.850 | 0.846 | 429 |
| 0.958 | 0.932 | 676 |
