# Milestone 2: rating candidates, game-level walk-forward backtest

Ratings are computed walk-forward over all games (both sources, team events included, walkovers skipped).
Hyperparameters tuned on 2015-2021 round-robin 3-way log loss; models compared on 2022-2024
(205,272 round-robin games with both players known, 178,422 with both seasoned 30+;
65,855 decided playoff games). `+var` = rating uncertainty integrated into the prediction.
Lower is better everywhere.

| model | tune_ll | select_seasoned_ll | select_seasoned_rps | select_all_ll | select_all_rps | select_all_po_ll |
|---|---|---|---|---|---|---|
| kalman+var | 0.7543 | 0.7769 | 0.1676 | 0.7606 | 0.1634 | 0.5766 |
| kalman | 0.7543 | 0.7769 | 0.1676 | 0.7606 | 0.1634 | 0.5766 |
| mov_elo | 0.7630 | 0.7789 | 0.1680 | 0.7682 | 0.1655 | 0.5766 |
| glicko2+var | 0.7644 | 0.7783 | 0.1678 | 0.7691 | 0.1657 | 0.5776 |
| glicko2 | 0.7644 | 0.7783 | 0.1678 | 0.7706 | 0.1660 | 0.5777 |
| elo | 0.7687 | 0.7834 | 0.1694 | 0.7735 | 0.1672 | 0.5820 |
| tw_bt+var | – | 0.7879 | 0.1706 | 0.7972 | 0.1750 | 0.5920 |
| tw_bt | – | 0.7902 | 0.1711 | 0.8007 | 0.1758 | 0.5944 |

Columns: `*_ll` 3-way log loss on round-robin games, `*_rps` ranked probability score, `*_po_ll` binary log loss
of the playoff game winner.

## Tuned parameters

```json
{
  "elo": {
    "k": 38.400000000000006,
    "k_new": 163.84000000000003,
    "new_games": 102.4,
    "mov": 0.0
  },
  "mov_elo": {
    "k": 30.720000000000006,
    "k_new": 163.84000000000003,
    "new_games": 102.4,
    "mov": 0.8
  },
  "glicko2": {
    "tau": 1.2800000000000002,
    "sigma0": 0.06,
    "phi0": 437.5,
    "c_day": 5.120000000000001,
    "phi_min": 30.0
  },
  "kalman": {
    "obs_sd": 1.2480000000000002,
    "prior_sd": 0.8999999999999999,
    "prior_mean": -0.8,
    "drift_sd_year": 0.5599999999999999,
    "drift_floor_day": 0.025,
    "junior_age": 20.0,
    "junior_mult": 2.0480000000000005,
    "gd_cap": 6.0,
    "level_prior": 0.5
  },
  "tw_bt": {
    "half_life_days": 219.0,
    "window_years": 4.0,
    "ridge": 1.2,
    "gd_cap": 6.0,
    "obs_sd": 2.4,
    "prior_mean": -0.8,
    "start": "2014-01-01"
  }
}
```

Runtime 2.6 min.
