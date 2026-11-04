import { useEffect, useMemo, useRef, useState } from "react";
import { EntryList } from "./components/EntryList";
import { FormatBuilder } from "./components/FormatBuilder";
import { ImportBox, importLink } from "./components/ImportBox";
import { PlayerPicker } from "./components/PlayerPicker";
import { Results } from "./components/Results";
import { gbmRequest, h2hFor, loadData, type AppData, type RankedPlayer } from "./data";
import type { Player, SimResult, StageConfig } from "./engine/types";
import type { ImportResult, RatingDefaults } from "./import/tableHockeyEu";
import { buildStages, defaultFormat, type FormatState } from "./formats";
import type { SimRequest } from "./worker";

const STORAGE_KEY = "thsim.state.v1";

interface Saved {
  selected: number[];
  format: FormatState;
  runs: number;
  uncertainty?: boolean;
  customJson?: string | null;
  /** A table-hockey.eu tournament link: reloaded on open, so shared links stay live. */
  importLink?: string | null;
}

/** A shared link (#s=...) wins over what this browser remembered. */
function loadSaved(): Partial<Saved> {
  try {
    const m = location.hash.match(/s=([^&]+)/);
    if (m) return JSON.parse(decodeURIComponent(escape(atob(decodeURIComponent(m[1])))));
  } catch {
    /* malformed link: fall back to local state */
  }
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "{}");
  } catch {
    return {};
  }
}

function shareUrl(state: Saved): string {
  const code = encodeURIComponent(btoa(unescape(encodeURIComponent(JSON.stringify(state)))));
  return `${location.origin}${location.pathname}#s=${code}`;
}

export function App() {
  const [data, setData] = useState<AppData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const saved = useMemo(loadSaved, []);
  const [selected, setSelected] = useState<number[]>(saved.selected ?? []);
  const [format, setFormat] = useState<FormatState>(saved.format ?? defaultFormat(16));
  const [formatTouched, setFormatTouched] = useState(!!saved.format);
  const [runs, setRuns] = useState(saved.runs ?? 10000);
  const [uncertainty, setUncertainty] = useState(saved.uncertainty ?? true);
  const [customJson, setCustomJson] = useState<string | null>(saved.customJson ?? null);
  const [progress, setProgress] = useState<number | null>(null);
  const [copied, setCopied] = useState(false);
  const [result, setResult] = useState<{ res: SimResult; stages: StageConfig[] } | null>(null);
  const [imported, setImported] = useState<{ link: string; result: ImportResult } | null>(null);
  const worker = useRef<Worker | null>(null);

  useEffect(() => {
    loadData().then(setData, (e) => setError(String(e)));
  }, []);

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({ selected, format, runs, uncertainty, customJson, importLink: imported?.link ?? null }));
    } catch {
      /* storage unavailable: nothing to remember */
    }
    setCopied(false);
  }, [selected, format, runs, uncertainty, customJson, imported]);

  const defaults: RatingDefaults = useMemo(() => {
    const rp = data?.meta.rating_params;
    return { prior_mean: rp?.prior_mean ?? -0.8, prior_sd: rp?.prior_sd ?? 0.9, level_prior: rp?.level_prior ?? 0.5 };
  }, [data]);

  const applyImport = (link: string, r: ImportResult) => {
    setImported({ link, result: r });
    setSelected(r.entries.map((p) => p.id));
    setCustomJson(JSON.stringify(r.stages, null, 2));
    setResult(null);
  };

  // Reopen a tournament that came with the link or was loaded last time.
  useEffect(() => {
    if (!data || !saved.importLink) return;
    importLink(saved.importLink, data.players, defaults).then(
      (r) => applyImport(saved.importLink!, r),
      () => undefined,
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data]);

  const byId = useMemo(
    () => new Map([...(data?.players ?? []), ...(imported?.result.entries ?? [])].map((p) => [p.id, p])),
    [data, imported],
  );
  const entries = selected.map((id) => byId.get(id)).filter((p): p is RankedPlayer => !!p);
  const n = entries.length;

  // Keep sensible defaults for the field size until the user edits the format.
  useEffect(() => {
    if (!formatTouched && n > 1) setFormat(defaultFormat(n));
  }, [n, formatTouched]);

  const built = useMemo(() => buildStages(format, n), [format, n]);
  let stages: StageConfig[] = built;
  let jsonError: string | null = null;
  if (customJson) {
    try {
      stages = JSON.parse(customJson);
    } catch (e) {
      jsonError = (e as Error).message;
    }
  }

  const toggle = (id: number) => setSelected((s) => (s.includes(id) ? s.filter((x) => x !== id) : [...s, id]));
  const addMany = (ids: number[]) => setSelected((s) => [...s, ...ids.filter((id) => !s.includes(id))]);

  const run = () => {
    if (!data || n < 2 || jsonError) return;
    worker.current?.terminate();
    const w = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
    worker.current = w;
    const players: Player[] = entries.map((p) => ({ id: p.id, name: p.name, country: p.country, mu: p.mu, sd: p.sd }));
    const req: SimRequest = {
      players, stages, model: data.model, runs, seed: Math.floor(Math.random() * 2 ** 31), ratingUncertainty: uncertainty,
      played: imported?.result.played,
      h2h: h2hFor(data.h2h, players.map((p) => p.id)),
      gbm: gbmRequest(data, entries, imported?.result.level ?? 0),
    };
    const runStages = stages;
    setProgress(0);
    setError(null);
    w.onmessage = (e) => {
      const m = e.data;
      if (m.type === "progress") setProgress(m.done / runs);
      else if (m.type === "done") {
        setResult({ res: m.result, stages: runStages });
        setProgress(null);
        w.terminate();
        requestAnimationFrame(() => document.getElementById("results")?.scrollIntoView({ behavior: "smooth" }));
      } else if (m.type === "error") {
        setError(m.message);
        setProgress(null);
      }
    };
    w.postMessage(req);
  };

  if (error && !data) return <div className="page"><p className="error">Could not load data: {error}</p></div>;
  if (!data) return <div className="page loading">Loading players and model…</div>;

  return (
    <div className="page">
      <header className="hero">
        <div>
          <h1>Table Hockey Tournament Simulator</h1>
          <p className="muted">
            Build a tournament from the world's players and see how it plays out over thousands of simulated runs. Ratings come
            from {data.meta.games.toLocaleString()} games up to {data.meta.latest_game}.
          </p>
        </div>
      </header>

      <ImportBox
        players={data.players}
        defaults={defaults}
        imported={imported}
        onImport={applyImport}
        onClear={() => {
          setImported(null);
          setCustomJson(null);
        }}
      />

      <div className="builder">
        <PlayerPicker
          players={data.players}
          selected={new Set(selected)}
          rankingAvailable={data.meta.ranking_available}
          onToggle={toggle}
          onAddMany={addMany}
        />
        <div className="side">
          <EntryList entries={entries} onChange={setSelected} />
          <FormatBuilder
            n={n}
            format={format}
            onChange={(f) => {
              setFormat(f);
              setFormatTouched(true);
            }}
            stages={built}
            customJson={customJson}
            onCustomJson={setCustomJson}
          />
          <section className="card run">
            <header className="card-head">
              <h2>
                <span className="step">4</span> Simulate
              </h2>
            </header>
            <div className="form-grid">
              <label>
                Runs
                <select value={runs} onChange={(e) => setRuns(+e.target.value)}>
                  {[1000, 10000, 50000, 100000].map((r) => (
                    <option key={r} value={r}>
                      {r.toLocaleString()}
                    </option>
                  ))}
                </select>
              </label>
              <label className="check" title="Draw each player's rating from its uncertainty once per simulated tournament">
                <input type="checkbox" checked={uncertainty} onChange={(e) => setUncertainty(e.target.checked)} /> Rating uncertainty
              </label>
            </div>
            {jsonError && <p className="error small">Config JSON: {jsonError}</p>}
            {error && <p className="error small">{error}</p>}
            <button className="primary" disabled={n < 2 || progress !== null || !!jsonError} onClick={run}>
              {progress !== null ? `Simulating… ${Math.round(progress * 100)}%` : n < 2 ? "Add at least 2 players" : `Simulate ${runs.toLocaleString()} tournaments`}
            </button>
            {progress !== null && (
              <div className="progress" aria-hidden>
                <span style={{ width: `${progress * 100}%` }} />
              </div>
            )}
            {n >= 2 && (
              <button
                className="link"
                onClick={() => {
                  const url = shareUrl({ selected, format, runs, uncertainty, customJson, importLink: imported?.link ?? null });
                  history.replaceState(null, "", url);
                  navigator.clipboard?.writeText(url).then(() => setCopied(true), () => setCopied(false));
                }}
              >
                {copied ? "Link copied" : "Copy a link to this tournament"}
              </button>
            )}
          </section>
        </div>
      </div>

      <div id="results">{result && <Results result={result.res} stages={result.stages} />}</div>

      <About data={data} />
    </div>
  );
}

function About({ data }: { data: AppData }) {
  return (
    <footer className="about card">
      <h3>How it works</h3>
      <p>
        Every player has a rating that tracks their skill over time, learned from{" "}
        {data.meta.games.toLocaleString()} games from th.sportscorpion.com and bordshockey.net. It blends a goal-difference
        rating with a Glicko-2 rating on a softened margin score, the method used by h2h.puck.no. Ratings drift faster for
        juniors and after long breaks, and the "±" shows how uncertain each one is. 100 rating points correspond to about one
        goal of expected margin per game.
      </p>
      {Object.keys(data.h2h).length > 0 && (
        <p>
          Some players match up better or worse against a particular opponent than their ratings suggest. When two players
          have met before, the games between them shift their head-to-head odds a little, shrunk toward zero until they have
          met often (also from h2h.puck.no).
        </p>
      )}
      {data.gbm && (
        <p>
          On top of the ratings, a gradient-boosted model (LightGBM) adjusts the odds of every pairing. It looks at several
          ratings at once, how many games each player has played, how often they draw, recent form and the head-to-head
          record. It helps most for players with few games behind them.
        </p>
      )}
      <p>
        During an event (a table-hockey.eu link with results), games already played are kept as they ended, and they also
        update every player's strength for the rest of the day: someone who has had a great group stage goes into the
        playoffs stronger than their rating alone says, and someone having a bad day weaker.
      </p>
      <p>
        Each simulated tournament draws every player's strength once from that uncertainty. Games are then played with a model
        fitted on real results: how often games are drawn at each strength gap, real scorelines (so goal difference tiebreaks
        behave), golden-goal overtime in playoffs (much closer to a coin flip than regulation), and first-to-N series. The
        model was chosen and checked with walk-forward backtests, which only ever use games played before the event being
        predicted.
      </p>
      <Backtest data={data} />
      <p className="muted small">
        Data generated {data.meta.generated}. Source code: github.com/amundfylling/th-simulator.
      </p>
    </footer>
  );
}

function Backtest({ data }: { data: AppData }) {
  const bt = data.meta.tournament_backtest;
  if (!bt) return null;
  const v = bt.variants;
  const previous = v["kalman_form0.45"] ?? v["kalman"];
  const ratingsOnly = v["composite+h2h_form0.45"];
  const model = (data.gbm ? v["gbm_form0.45"] : undefined) ?? ratingsOnly ?? previous;
  if (!model) return null;
  const gm = (ll: number) => `${(Math.exp(-ll) * 100).toFixed(1)}%`;
  type Row = [string, { winner_ll: number; fav_won: number; rps: number } | undefined, string];
  const before: Row[] = [];
  if (ratingsOnly && model !== ratingsOnly) before.push(["Ratings + head-to-head only", ratingsOnly, "without the LightGBM layer"]);
  if (previous && model !== previous) before.push(["Goal-difference rating only", previous, "the original model"]);
  const rows: Row[] = [
    ["This simulator", model, "ratings with uncertainty, form, real formats"],
    ...before,
    ["Plain Elo, same engine", v["elo"], "no uncertainty"],
    ["Everyone equal", { winner_ll: model.uniform_ll, fav_won: NaN, rps: NaN }, "baseline"],
  ];
  return (
    <>
      <h3>How well does it predict?</h3>
      <p>
        Tested on {model.events} real tournaments from {bt.window[0].slice(0, 4)}–{bt.window[1].slice(0, 4)}, each simulated
        using only games played before it. The actual winner's predicted chance (geometric mean) and how often the
        pre-tournament favourite won:
      </p>
      <div className="table-scroll">
        <table className="bt">
          <thead>
            <tr>
              <th>Model</th>
              <th className="num">Winner's chance</th>
              <th className="num">Favourite won</th>
              <th className="hide-sm">Notes</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(([name, r, note]) =>
              r ? (
                <tr key={name}>
                  <td>{name}</td>
                  <td className="num">{gm(r.winner_ll)}</td>
                  <td className="num">{Number.isNaN(r.fav_won) ? "–" : `${Math.round(r.fav_won * 100)}%`}</td>
                  <td className="muted hide-sm">{note}</td>
                </tr>
              ) : null,
            )}
          </tbody>
        </table>
      </div>
    </>
  );
}
