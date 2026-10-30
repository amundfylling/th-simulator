# th-simulator

Monte Carlo table hockey tournament simulator: for every player, the probability of
finishing 1st, 2nd, 3rd … Nth, validated by walk-forward backtests on historical
tournaments. The end product is a website where you build a tournament from the ITHF
world ranking and see the simulated outcome.

## Layout

| Path | What |
|---|---|
| `thsim/data/sources.py` | Public data URLs, downloaded at runtime into `data_cache/` (git-ignored) |
| `thsim/data/loading.py` | Load and normalise Scorpion + bordshockey.net games, cross-source dedupe, players, tournaments, ITHF ranking |
| `thsim/structure.py` | Stage-structure inference: groups (connected components), playoff series, wins needed per round, top places |
| `thsim/ratings/elo.py` | Baseline walk-forward Elo |
| `thsim/models/ordered_logit.py` | Three-way regulation outcome model |
| `thsim/metrics.py` | Log loss, Brier, RPS, reliability |
| `thsim/facts.py` | Reproduces the known data facts (`reports/m1_facts.md`) |

## Usage

```bash
uv venv -p 3.12 && uv pip install -e '.[dev]'
python -m thsim.facts                 # downloads data, writes reports/m1_facts.md
python -m thsim.facts --scorpion-only # same on Scorpion games alone
pytest
```

Data sources can be redirected with `THSIM_URL_<NAME>` environment variables and the
cache folder with `THSIM_CACHE`.
