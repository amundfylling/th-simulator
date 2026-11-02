# th-simulator

Monte Carlo table hockey tournament simulator. Pick players from the ITHF world ranking,
choose a format, and see each player's probability of finishing 1st, 2nd, 3rd … Nth.
The model behind it is validated by walk-forward backtests on historical games and
tournaments.

```
Python (thsim/)                                  TypeScript (web/)
  data: Scorpion + bordshockey.net games  ──►     engine/: tournament simulator
  ratings: Kalman goal-difference filter          (groups, Swiss, knockout series,
  game model: draw / OT / scoreline               byes, placement brackets, live mode)
  backtests: game, series, tournament    ──JSON─►  website: picker, format builder,
  export.py → web/public/data/*.json              heatmaps, road to the title
```

The same engine runs on the website (in a Web Worker) and in the tournament backtest
(Node), so the odds on the site come from the code that was validated.

## Layout

| Path | What |
|---|---|
| `thsim/data/` | Download (runtime, never committed), normalise and de-duplicate the sources |
| `thsim/structure.py` | Stage-structure inference: groups, playoff series, wins needed per round, top places |
| `thsim/ratings/` | Rating candidates: Elo, margin-of-victory Elo, Glicko-2, time-weighted Bradley–Terry, Kalman goal-difference filter |
| `thsim/models/` | Ordered logit and the full game model (regulation, overtime, scorelines) |
| `thsim/backtest_games.py` | Game-level walk-forward comparison of rating models (M2) |
| `thsim/fit_game_model.py` | Game, series and shape checks for the game model (M3) |
| `thsim/backtest_tournaments.py` | Tournament reconstruction and tournament-level backtest (M5) |
| `thsim/export.py` | Current ratings + game model for the website |
| `web/src/engine/` | Tournament engine (TypeScript) |
| `web/src/` | Website (React + Vite) |
| `reports/` | Backtest reports for each milestone |

## Usage

```bash
# Python
uv venv -p 3.12 && uv pip install -e '.[dev]'
python -m thsim.facts                     # M1 data facts
python -m thsim.backtest_games            # M2 rating comparison (~15 min)
python -m thsim.fit_game_model            # M3 game model report
python -m thsim.backtest_tournaments all  # M5 tournament backtest (needs web/ deps)
python -m thsim.export                    # write web/public/data/
pytest

# Website
cd web && npm ci
npm run dev        # local site (run thsim.export first)
npm test           # engine tests
npm run build      # static site in web/dist
```

A tournament config is a list of stages. Each stage takes entrants from the entry list
or from slices of earlier stages' rankings, and `places` fixes final places:

```json
[
  {"id": "groups", "type": "groups", "groups": 4, "legs": 1, "source": [{"from": "entry"}], "places": 1},
  {"id": "ko", "type": "knockout", "winsNeeded": [3, 4, 4], "bronze": true,
   "source": [{"from": "groups", "ranks": [1, 8]}], "places": 1},
  {"id": "b", "type": "knockout", "winsNeeded": [2], "source": [{"from": "groups", "ranks": [9, 16]}], "places": 9}
]
```

Games already played (live mode) go in `played: [{stage, p1, p2, g1, g2, ot}]`.

On the website, a table-hockey.eu tournament link (e.g. `https://table-hockey.eu/tournaments/33`)
loads the entry list, the real stages (groups, playoff series lengths, bronze, which places feed
which bracket) and every game played so far from its public API, then simulates what is still
open. Players are matched by ITHF id, then by name; anyone without rating history starts as a
newcomer for the event's level. table-hockey.eu does not let other sites read its API, so the
site goes through public CORS relays. For a dependable relay of your own, deploy
`proxy/worker.js` as a free Cloudflare Worker and set the repository variable `TH_PROXY` to
`https://<worker>.workers.dev/?url=`; the nightly build picks it up.

## Deployment

`.github/workflows/site.yml` rebuilds the ratings and the site every night and publishes
to GitHub Pages (enable Pages with source "GitHub Actions" in the repository settings).
