# Milestone 2: rating candidates, game-level walk-forward backtest

Ratings are computed walk-forward over all games (both sources, team events included, walkovers skipped).
Hyperparameters tuned on 2015-2021 round-robin 3-way log loss; models compared on 2022-2024
(205,272 round-robin games with both players known, 178,445 with both seasoned 30+;
65,855 decided playoff games). `+var` = rating uncertainty integrated into the prediction.
Lower is better everywhere.

| model | tune_ll | select_seasoned_ll | select_seasoned_rps | select_all_ll | select_all_rps | select_all_po_ll | final_seasoned_ll | final_all_ll | final_all_po_ll |
|---|---|---|---|---|---|---|---|---|---|
| kalman+var | nan | 0.7768 | 0.1675 | 0.7606 | 0.1634 | 0.5766 | 0.7813 | 0.7689 | 0.5890 |
| kalman | nan | 0.7769 | 0.1675 | 0.7606 | 0.1634 | 0.5766 | 0.7813 | 0.7688 | 0.5890 |
| mov_elo | nan | 0.7788 | 0.1680 | 0.7682 | 0.1655 | 0.5766 | 0.7813 | 0.7728 | 0.5885 |
| glicko2+var | nan | 0.7782 | 0.1677 | 0.7691 | 0.1657 | 0.5776 | 0.7793 | 0.7721 | 0.5897 |
| glicko2 | nan | 0.7782 | 0.1678 | 0.7706 | 0.1660 | 0.5777 | 0.7793 | 0.7729 | 0.5899 |
| elo | nan | 0.7833 | 0.1694 | 0.7735 | 0.1672 | 0.5820 | 0.7852 | 0.7773 | 0.5934 |
| tw_bt+var | nan | 0.7834 | 0.1691 | 0.7860 | 0.1712 | 0.5887 | 0.7869 | 0.7876 | 0.6027 |
| tw_bt | nan | 0.7851 | 0.1696 | 0.7888 | 0.1719 | 0.5903 | 0.7886 | 0.7901 | 0.6040 |

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

Runtime 2.3 min.
