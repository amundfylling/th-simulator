import { useMemo, useState } from "react";
import { assignGroups } from "../engine/simulate";
import type { SimResult, StageConfig } from "../engine/types";
import { displayRating } from "../data";
import { SingleRun } from "./SingleRun";

interface Props {
  result: SimResult;
  stages: StageConfig[];
}

interface Row {
  i: number;
  id: number;
  name: string;
  country?: string;
  seed: number;
  p: number[];
  expPlace: number;
}

const pct = (p: number, digits = 1) => (p >= 0.9995 ? "100%" : p < 0.0005 ? "<0.1%" : `${(p * 100).toFixed(digits)}%`);
const se = (p: number, runs: number) => Math.sqrt((p * (1 - p)) / runs);

/** Sequential blue ramp (light→dark); the lightest steps recede toward the surface. */
const RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"];
function heat(p: number): string | undefined {
  if (p < 0.001) return undefined;
  const t = Math.min(1, Math.pow(p / 0.6, 0.55));
  return RAMP[Math.min(RAMP.length - 1, Math.floor(t * RAMP.length))];
}
const inkOn = (p: number) => (Math.pow(Math.min(1, p / 0.6), 0.55) > 0.45 ? "#ffffff" : "var(--text-primary)");

export function Results({ result, stages }: Props) {
  if (result.runs === 1 && result.single) return <SingleRun result={result} stages={stages} />;
  return <Distribution result={result} stages={stages} />;
}

function Distribution({ result, stages }: Props) {
  const R = result.runs;
  const n = result.players.length;
  const [tip, setTip] = useState<{ x: number; y: number; text: string } | null>(null);
  const [sortKey, setSortKey] = useState<"exp" | "win" | "seed">("exp");

  const rows: Row[] = useMemo(
    () =>
      result.players.map((pl, i) => {
        const p = result.placeCounts[i].map((c) => c / R);
        return { i, id: pl.id, name: pl.name, country: pl.country, seed: i + 1, p, expPlace: p.reduce((s, v, k) => s + v * (k + 1), 0) };
      }),
    [result, R],
  );
  const sorted = useMemo(() => {
    const r = rows.slice();
    if (sortKey === "win") r.sort((a, b) => b.p[0] - a.p[0] || a.expPlace - b.expPlace);
    else if (sortKey === "seed") r.sort((a, b) => a.seed - b.seed);
    else r.sort((a, b) => a.expPlace - b.expPlace);
    return r;
  }, [rows, sortKey]);

  const byWin = useMemo(() => rows.slice().sort((a, b) => b.p[0] - a.p[0]), [rows]);
  const columns = useMemo(() => placeColumns(n), [n]);
  const finals = useMemo(
    () =>
      Object.entries(result.finals)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 8)
        .map(([key, c]) => {
          const [a, b] = key.split("-").map(Number);
          const name = (id: number) => result.players.find((p) => p.id === id)?.name ?? String(id);
          return { a: name(a), b: name(b), p: c / R };
        }),
    [result, R],
  );
  const fav = byWin[0];
  const topSeedWin = rows[0]?.p[0] ?? 0;
  const outsider = rows.slice(4).reduce((s, r) => s + r.p[0], 0);

  const showTip = (e: React.MouseEvent, text: string) => setTip({ x: e.clientX, y: e.clientY, text });

  return (
    <div className="results" onMouseLeave={() => setTip(null)}>
      <div className="tiles">
        <Tile label="Favourite" value={fav.name} sub={`${pct(fav.p[0])} to win`} />
        <Tile label="Top seed wins" value={pct(topSeedWin)} sub={rows[0].name} />
        <Tile label="Most likely final" value={finals[0] ? `${finals[0].a} – ${finals[0].b}` : "–"} sub={finals[0] ? pct(finals[0].p) : ""} />
        <Tile label="Winner outside top 4 seeds" value={pct(outsider)} sub={`${R.toLocaleString()} simulated tournaments`} />
      </div>

      <section className="card">
        <header className="card-head">
          <h3>Who wins?</h3>
          <p className="muted small">Probability of winning the tournament, with Monte Carlo error (±1 s.e.).</p>
        </header>
        <div className="bars">
          {byWin.slice(0, 16).map((r) => (
            <div className="bar-row" key={r.id} onMouseMove={(e) => showTip(e, `${r.name}: ${pct(r.p[0], 2)} ± ${pct(se(r.p[0], R), 2)} · seed ${r.seed}`)}>
              <span className="bar-label">
                <span className="muted small">{r.seed}.</span> {r.name}
              </span>
              <span className="bar-track">
                <span className="bar-fill" style={{ width: `${(r.p[0] / Math.max(byWin[0].p[0], 1e-9)) * 100}%` }} />
              </span>
              <span className="bar-value">
                {pct(r.p[0])} <span className="muted small">±{(se(r.p[0], R) * 100).toFixed(1)}</span>
              </span>
            </div>
          ))}
        </div>
      </section>

      <section className="card">
        <header className="card-head row">
          <div>
            <h3>Finishing places</h3>
            <p className="muted small">Each row is a player, each column a final place. Darker means more likely. Hover a cell for the exact number.</p>
          </div>
          <div className="segmented small">
            {(["exp", "win", "seed"] as const).map((k) => (
              <button key={k} className={sortKey === k ? "on" : ""} onClick={() => setSortKey(k)}>
                {k === "exp" ? "Expected place" : k === "win" ? "Win chance" : "Seed"}
              </button>
            ))}
          </div>
        </header>
        <div className="table-scroll heat-scroll">
          <table className="heat">
            <thead>
              <tr>
                <th className="sticky">Player</th>
                <th className="num">Exp.</th>
                {columns.map((c) => (
                  <th key={c.label} className="num">
                    {c.label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {sorted.map((r) => (
                <tr key={r.id}>
                  <td className="sticky">
                    <span className="muted small">{r.seed}.</span> {r.name}
                  </td>
                  <td className="num muted">{r.expPlace.toFixed(1)}</td>
                  {columns.map((c) => {
                    const v = r.p.slice(c.from, c.to + 1).reduce((s, x) => s + x, 0);
                    return (
                      <td
                        key={c.label}
                        className="cell"
                        style={{ background: heat(v), color: inkOn(v) }}
                        onMouseMove={(e) => showTip(e, `${r.name} · place ${c.label}: ${pct(v, 2)} ± ${pct(se(v, R), 2)}`)}
                      >
                        {v >= 0.01 ? Math.round(v * 100) : ""}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <Legend />
      </section>

      {stages.map((s, si) => {
        const st = result.stages[si];
        if (s.type === "knockout" && st.reached) return <Road key={s.id} title={s.name ?? "Playoff"} result={result} si={si} showTip={showTip} />;
        if (s.type === "groups" && st.position)
          return <Groups key={s.id} stage={s} stages={stages} result={result} si={si} showTip={showTip} />;
        return null;
      })}

      {finals.length > 0 && (
        <section className="card">
          <header className="card-head">
            <h3>Most likely finals</h3>
          </header>
          <div className="bars">
            {finals.map((f) => (
              <div className="bar-row" key={f.a + f.b}>
                <span className="bar-label">
                  {f.a} – {f.b}
                </span>
                <span className="bar-track">
                  <span className="bar-fill alt" style={{ width: `${(f.p / finals[0].p) * 100}%` }} />
                </span>
                <span className="bar-value">{pct(f.p)}</span>
              </div>
            ))}
          </div>
        </section>
      )}

      <div className="downloads">
        <button className="chip" onClick={() => download("place-probabilities.csv", toCsv(rows, R), "text/csv")}>
          Download CSV
        </button>
        <button className="chip" onClick={() => download("simulation.json", JSON.stringify(toJson(rows, result), null, 1), "application/json")}>
          Download JSON
        </button>
        <span className="muted small">Simulated in {(result.elapsedMs / 1000).toFixed(1)} s.</span>
      </div>
      {tip && (
        <div className="tooltip" style={{ left: tip.x + 14, top: tip.y + 14 }} role="status">
          {tip.text}
        </div>
      )}
    </div>
  );
}

function Tile({ label, value, sub }: { label: string; value: string; sub: string }) {
  return (
    <div className="tile">
      <span className="tile-label">{label}</span>
      <span className="tile-value">{value}</span>
      <span className="tile-sub muted">{sub}</span>
    </div>
  );
}

function Legend() {
  return (
    <div className="legend">
      <span className="muted small">less likely</span>
      {[0.002, 0.01, 0.03, 0.08, 0.15, 0.3, 0.6].map((p) => (
        <span key={p} className="swatch" style={{ background: heat(p) }} title={pct(p)} />
      ))}
      <span className="muted small">more likely</span>
      <span className="muted small">· numbers are percent</span>
    </div>
  );
}

function Road({ title, result, si, showTip }: { title: string; result: SimResult; si: number; showTip: (e: React.MouseEvent, t: string) => void }) {
  const st = result.stages[si];
  const R = result.runs;
  const reached = st.reached!;
  // Only rounds that someone actually played in.
  const used = st.roundNames!.map((_, r) => reached.some((row) => row[r] > 0));
  const first = used.indexOf(true);
  const names = st.roundNames!.slice(first);
  const players = result.players
    .map((p, i) => ({ p, i, entered: st.entered[i] / R, win: reached[i][reached[i].length - 1] / R }))
    .filter((x) => x.entered > 0)
    .sort((a, b) => b.win - a.win || b.entered - a.entered)
    .slice(0, 40);
  return (
    <section className="card">
      <header className="card-head">
        <h3>{title}: road to the title</h3>
        <p className="muted small">Probability of reaching each round, including byes.</p>
      </header>
      <div className="table-scroll">
        <table className="road">
          <thead>
            <tr>
              <th>Player</th>
              {names.map((n) => (
                <th key={n} className="num">
                  {n}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {players.map(({ p, i }) => (
              <tr key={p.id}>
                <td>{p.name}</td>
                {names.map((n, k) => {
                  const v = reached[i][first + k] / R;
                  return (
                    <td key={n} className="num minibar" onMouseMove={(e) => showTip(e, `${p.name} reaches ${n}: ${pct(v, 2)}`)}>
                      <span className="mb" style={{ width: `${v * 100}%` }} />
                      <span className="mv">{pct(v)}</span>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function Groups({ stage, stages, result, si, showTip }: { stage: StageConfig & { type: "groups" }; stages: StageConfig[]; result: SimResult; si: number; showTip: (e: React.MouseEvent, t: string) => void }) {
  const st = result.stages[si];
  const R = result.runs;
  // Group membership is fixed when the stage seeds straight from the entry list.
  const fromEntry = stage.source.length === 1 && stage.source[0].from === "entry" && !stage.source[0].ranks && stage.seeding !== "random";
  const idx = result.players.map((_, i) => i);
  const G = Math.max(1, Math.min(stage.groups, idx.length));
  const groups = fromEntry ? assignGroups(idx, G, stage.seeding === "block" ? "block" : "snake") : [idx.filter((i) => st.entered[i] > 0)];
  const next = stages.slice(si + 1).filter((s) => s.source.some((src) => src.from === stage.id && (src.ranks?.[0] ?? 1) === 1));
  const nextIdx = next.length ? stages.indexOf(next[0]) : -1;
  return (
    <section className="card">
      <header className="card-head">
        <h3>{stage.name ?? "Groups"}</h3>
        <p className="muted small">
          Chance of each group position{nextIdx >= 0 ? ", of reaching the playoff" : ""}, and expected points.
        </p>
      </header>
      <div className="groups">
        {groups.map((members, g) => {
          const maxPos = members.length;
          const rows = members
            .map((i) => ({ i, pos: st.position![i].slice(0, maxPos).map((c) => c / R), adv: nextIdx >= 0 ? result.stages[nextIdx].entered[i] / R : null, pts: st.pointsSum![i] / Math.max(1, st.entered[i]) }))
            .sort((a, b) => b.pts - a.pts);
          return (
            <div className="group" key={g}>
              <h4>{groups.length > 1 ? `Group ${String.fromCharCode(65 + g)}` : "Table"}</h4>
              <table className="heat compact">
                <thead>
                  <tr>
                    <th>Player</th>
                    {Array.from({ length: maxPos }, (_, k) => (
                      <th key={k} className="num">
                        {k + 1}
                      </th>
                    ))}
                    {nextIdx >= 0 && <th className="num">Adv.</th>}
                    <th className="num">Pts</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r) => {
                    const p = result.players[r.i];
                    return (
                      <tr key={p.id}>
                        <td title={`Rating ${displayRating(p.mu)}`}>{p.name}</td>
                        {r.pos.map((v, k) => (
                          <td key={k} className="cell" style={{ background: heat(v), color: inkOn(v) }} onMouseMove={(e) => showTip(e, `${p.name} finishes ${k + 1} in the group: ${pct(v, 2)}`)}>
                            {v >= 0.01 ? Math.round(v * 100) : ""}
                          </td>
                        ))}
                        {r.adv !== null && <td className="num strong">{pct(r.adv, 0)}</td>}
                        <td className="num muted">{r.pts.toFixed(1)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          );
        })}
      </div>
    </section>
  );
}

function placeColumns(n: number): { label: string; from: number; to: number }[] {
  if (n <= 24) return Array.from({ length: n }, (_, k) => ({ label: String(k + 1), from: k, to: k }));
  const cols = [0, 1, 2, 3].map((k) => ({ label: String(k + 1), from: k, to: k }));
  let lo = 5;
  while (lo <= n) {
    const hi = Math.min(n, (lo - 1) * 2);
    cols.push({ label: lo === hi ? String(lo) : `${lo}-${hi}`, from: lo - 1, to: hi - 1 });
    lo = hi + 1;
  }
  return cols;
}

function toCsv(rows: Row[], runs: number): string {
  const n = rows.length;
  const head = ["seed", "player_id", "name", "expected_place", ...Array.from({ length: n }, (_, k) => `p_place_${k + 1}`), ...Array.from({ length: n }, (_, k) => `se_place_${k + 1}`)];
  const lines = rows.map((r) =>
    [r.seed, r.id, JSON.stringify(r.name), r.expPlace.toFixed(3), ...r.p.map((v) => v.toFixed(5)), ...r.p.map((v) => se(v, runs).toFixed(5))].join(","),
  );
  return [head.join(","), ...lines].join("\n");
}

function toJson(rows: Row[], result: SimResult) {
  return {
    runs: result.runs,
    players: rows.map((r) => ({ seed: r.seed, id: r.id, name: r.name, expected_place: r.expPlace, p_place: r.p, se_place: r.p.map((v) => se(v, result.runs)) })),
    finals: result.finals,
  };
}

function download(name: string, text: string, type: string) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
