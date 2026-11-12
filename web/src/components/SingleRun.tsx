import { useMemo } from "react";
import type { GameRecord, SimResult, StageConfig, StageOutcome, TableRow } from "../engine/types";

interface Props {
  result: SimResult;
  stages: StageConfig[];
}

const ordinal = (k: number) => {
  const s = ["th", "st", "nd", "rd"];
  const v = k % 100;
  return k + (s[(v - 20) % 10] || s[v] || s[0]);
};

/** One simulated tournament, shown as it played out: standings, tables, brackets and every score. */
export function SingleRun({ result, stages }: Props) {
  const single = result.single!;
  const players = result.players;
  const name = (i: number) => players[i]?.name ?? "?";
  const standings = useMemo(
    () => players.map((p, i) => ({ i, p, place: single.places[i] + 1 })).sort((a, b) => a.place - b.place),
    [players, single],
  );
  const at = (place: number) => standings.find((s) => s.place === place);
  const topSeedPlace = single.places[0] + 1;

  return (
    <div className="results">
      <div className="tiles">
        <Tile label="Champion" value={at(1) ? name(at(1)!.i) : "–"} sub={at(1) ? `seed ${at(1)!.i + 1}` : ""} />
        <Tile label="Runner-up" value={at(2) ? name(at(2)!.i) : "–"} sub={at(2) ? `seed ${at(2)!.i + 1}` : ""} />
        <Tile label="Third" value={at(3) ? name(at(3)!.i) : "–"} sub={at(3) ? `seed ${at(3)!.i + 1}` : ""} />
        <Tile label="Top seed finished" value={ordinal(topSeedPlace)} sub={`${name(0)} · one simulated tournament`} />
      </div>

      {single.stages.map((o, si) => {
        const cfg = stages[si];
        const title = cfg?.name ?? (o.type === "knockout" ? "Playoff" : o.type === "swiss" ? "Swiss" : "Groups");
        if (o.type === "knockout") return <Bracket key={o.id} title={title} outcome={o} name={name} />;
        if (o.type === "groups") return <GroupTables key={o.id} title={title} outcome={o} name={name} />;
        return <SwissTable key={o.id} title={title} outcome={o} name={name} />;
      })}

      <section className="card">
        <header className="card-head">
          <h3>Final standings</h3>
        </header>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th className="num">Place</th>
                <th>Player</th>
                <th className="num">Seed</th>
              </tr>
            </thead>
            <tbody>
              {standings.map((s) => (
                <tr key={s.p.id}>
                  <td className="num strong">{s.place}</td>
                  <td>{s.p.name}</td>
                  <td className="num muted">{s.i + 1}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <div className="downloads">
        <button className="chip" onClick={() => download("tournament.json", JSON.stringify(toJson(result), null, 1), "application/json")}>
          Download JSON
        </button>
        <span className="muted small">Run again for a different outcome, or pick more runs to see the odds.</span>
      </div>
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

const score = (g: GameRecord) => `${g.g1}–${g.g2}${g.ot ? " OT" : ""}`;

function Standings({ rows, name }: { rows: (TableRow & { draws: number })[]; name: (i: number) => string }) {
  return (
    <table className="compact-table">
      <thead>
        <tr>
          <th className="num">#</th>
          <th>Player</th>
          <th className="num">GP</th>
          <th className="num">W</th>
          <th className="num">D</th>
          <th className="num">L</th>
          <th className="num">Goals</th>
          <th className="num">Pts</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r, k) => {
          return (
            <tr key={r.p}>
              <td className="num muted">{k + 1}</td>
              <td className="name-cell">{name(r.p)}</td>
              <td className="num">{r.games}</td>
              <td className="num">{r.wins}</td>
              <td className="num">{r.draws}</td>
              <td className="num">{r.games - r.wins - r.draws}</td>
              <td className="num">
                {r.gf}–{r.ga}
              </td>
              <td className="num strong">{r.points}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function GameList({ games, name }: { games: GameRecord[]; name: (i: number) => string }) {
  return (
    <ul className="game-list">
      {games.map((g, k) => (
        <li key={k}>
          <span className={g.g1 > g.g2 ? "strong" : ""}>{name(g.a)}</span> <span className="score">{score(g)}</span>{" "}
          <span className={g.g2 > g.g1 ? "strong" : ""}>{name(g.b)}</span>
        </li>
      ))}
    </ul>
  );
}

function byLabel(games: GameRecord[]): [string, GameRecord[]][] {
  const out = new Map<string, GameRecord[]>();
  for (const g of games) {
    const base = g.label.replace(" (carried over)", "");
    if (!out.has(base)) out.set(base, []);
    out.get(base)!.push(g);
  }
  return [...out.entries()];
}

function GroupTables({ title, outcome, name }: { title: string; outcome: StageOutcome; name: (i: number) => string }) {
  const tables = withDraws(outcome, outcome.tables ?? []);
  const games = byLabel(outcome.games);
  return (
    <section className="card">
      <header className="card-head">
        <h3>{title}</h3>
      </header>
      <div className="groups">
        {tables.map((rows, g) => {
          const label = tables.length > 1 ? `Group ${String.fromCharCode(65 + g)}` : "Table";
          const gs = games.find(([l]) => l === label)?.[1] ?? [];
          return (
            <div className="group" key={g}>
              <h4>{label}</h4>
              <Standings rows={rows} name={name} />
              {gs.length > 0 && (
                <details className="games">
                  <summary className="small muted">{gs.length} games</summary>
                  <GameList games={gs} name={name} />
                </details>
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}

function SwissTable({ title, outcome, name }: { title: string; outcome: StageOutcome; name: (i: number) => string }) {
  const [rows] = withDraws(outcome, outcome.tables ?? []);
  const rounds = byLabel(outcome.games);
  return (
    <section className="card">
      <header className="card-head">
        <h3>{title}</h3>
      </header>
      <div className="table-scroll">{rows && <Standings rows={rows} name={name} />}</div>
      {rounds.length > 0 && (
        <details className="games">
          <summary className="small muted">Games by round</summary>
          <div className="groups">
            {rounds.map(([label, gs]) => (
              <div key={label}>
                <h4>{label}</h4>
                <GameList games={gs} name={name} />
              </div>
            ))}
          </div>
        </details>
      )}
    </section>
  );
}

interface Series {
  a: number;
  b: number;
  games: GameRecord[];
  winsA: number;
  winsB: number;
}

function Bracket({ title, outcome, name }: { title: string; outcome: StageOutcome; name: (i: number) => string }) {
  const rounds = byLabel(outcome.games).map(([label, gs]) => {
    const series: Series[] = [];
    for (const g of gs) {
      let s = series.find((x) => x.a === g.a && x.b === g.b);
      if (!s) series.push((s = { a: g.a, b: g.b, games: [], winsA: 0, winsB: 0 }));
      s.games.push(g);
      if (g.g1 > g.g2) s.winsA++;
      else if (g.g2 > g.g1) s.winsB++;
    }
    return { label, series };
  });
  const champion = outcome.ranking[0];
  return (
    <section className="card">
      <header className="card-head">
        <h3>{title}</h3>
        {champion !== undefined && <p className="muted small">Won by {name(champion)}.</p>}
      </header>
      <div className="bracket">
        {rounds.map(({ label, series }) => (
          <div className="bracket-round" key={label}>
            <h4>{label}</h4>
            <div className="bracket-matches">
            {series.map((s, k) => {
              const aWon = s.winsA > s.winsB;
              return (
                <div className="match" key={k}>
                  <div className={`match-side${aWon ? " won" : ""}`}>
                    <span className="match-name">{name(s.a)}</span>
                    <span className="num">{s.games.length > 1 ? s.winsA : s.games[0].g1}</span>
                  </div>
                  <div className={`match-side${!aWon ? " won" : ""}`}>
                    <span className="match-name">{name(s.b)}</span>
                    <span className="num">{s.games.length > 1 ? s.winsB : s.games[0].g2}</span>
                  </div>
                  {(s.games.length > 1 || s.games[0].ot) && (
                    <div className="match-games muted small">{s.games.map(score).join(", ")}</div>
                  )}
                </div>
              );
            })}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

/** Adds draws to each table row, counted from the stage's games. */
function withDraws(outcome: StageOutcome, tables: TableRow[][]): (TableRow & { draws: number })[][] {
  const draws = new Map<number, number>();
  for (const g of outcome.games) {
    if (g.g1 !== g.g2) continue;
    draws.set(g.a, (draws.get(g.a) ?? 0) + 1);
    draws.set(g.b, (draws.get(g.b) ?? 0) + 1);
  }
  return tables.map((rows) => rows.map((r) => ({ ...r, draws: draws.get(r.p) ?? 0 })));
}

function toJson(result: SimResult) {
  const single = result.single!;
  const id = (i: number) => result.players[i].id;
  return {
    runs: 1,
    standings: result.players
      .map((p, i) => ({ place: single.places[i] + 1, seed: i + 1, id: p.id, name: p.name }))
      .sort((a, b) => a.place - b.place),
    stages: single.stages.map((o) => ({
      id: o.id,
      type: o.type,
      ranking: o.ranking.map(id),
      tables: o.tables?.map((t) => t.map((r) => ({ ...r, p: id(r.p) }))),
      games: o.games.map((g) => ({ ...g, a: id(g.a), b: id(g.b) })),
    })),
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
