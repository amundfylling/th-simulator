import { useMemo, useState } from "react";
import { displayRating, displaySd, type RankedPlayer } from "../data";

interface Props {
  players: RankedPlayer[];
  selected: Set<number>;
  rankingAvailable: boolean;
  onToggle: (id: number) => void;
  onAddMany: (ids: number[]) => void;
}

const PAGE = 150;

export function PlayerPicker({ players, selected, rankingAvailable, onToggle, onAddMany }: Props) {
  const [query, setQuery] = useState("");
  const [country, setCountry] = useState("");
  const [onlyRanked, setOnlyRanked] = useState(rankingAvailable);
  const [limit, setLimit] = useState(PAGE);

  const countries = useMemo(() => {
    const c = new Map<string, number>();
    for (const p of players) if (p.country) c.set(p.country, (c.get(p.country) ?? 0) + 1);
    return [...c.entries()].sort((a, b) => b[1] - a[1]).map(([k]) => k);
  }, [players]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return players.filter(
      (p) =>
        (!q || p.name.toLowerCase().includes(q)) &&
        (!country || p.country === country) &&
        (!onlyRanked || p.worldRank !== null),
    );
  }, [players, query, country, onlyRanked]);

  const addTop = (n: number) => onAddMany(filtered.slice(0, n).map((p) => p.id));

  return (
    <section className="card picker">
      <header className="card-head">
        <h2>
          <span className="step">1</span> Pick players
        </h2>
        <p className="muted">
          {rankingAvailable ? "ITHF world ranking order." : "Sorted by rating (the world ranking loads on the deployed site)."} Click a
          row to add or remove it.
        </p>
      </header>
      <div className="filters">
        <input
          type="search"
          placeholder="Search name…"
          value={query}
          onChange={(e) => {
            setQuery(e.target.value);
            setLimit(PAGE);
          }}
          aria-label="Search players"
        />
        <select value={country} onChange={(e) => setCountry(e.target.value)} aria-label="Country">
          <option value="">All countries</option>
          {countries.map((c) => (
            <option key={c}>{c}</option>
          ))}
        </select>
        {rankingAvailable && (
          <label className="check">
            <input type="checkbox" checked={onlyRanked} onChange={(e) => setOnlyRanked(e.target.checked)} /> Ranked only
          </label>
        )}
      </div>
      <div className="quick">
        Add top
        {[8, 16, 32, 64].map((n) => (
          <button key={n} className="chip" onClick={() => addTop(n)}>
            {n}
          </button>
        ))}
        <span className="muted">of {filtered.length.toLocaleString()} shown</span>
      </div>
      <div className="table-scroll">
        <table className="players">
          <thead>
            <tr>
              <th className="num">{rankingAvailable ? "WR" : "#"}</th>
              <th>Player</th>
              <th>Country</th>
              <th className="num" title="1500 + 100 per goal of expected margin; ± is the rating uncertainty">
                Rating
              </th>
              <th className="num hide-sm">Games</th>
              <th className="hide-sm">Last game</th>
            </tr>
          </thead>
          <tbody>
            {filtered.slice(0, limit).map((p, i) => (
              <tr key={p.id} className={selected.has(p.id) ? "sel" : ""} onClick={() => onToggle(p.id)}>
                <td className="num">{p.worldRank ?? (rankingAvailable ? "–" : i + 1)}</td>
                <td>
                  <span className="tick" aria-hidden>
                    {selected.has(p.id) ? "✓" : "+"}
                  </span>
                  {p.name}
                </td>
                <td>{p.country}</td>
                <td className="num">
                  {displayRating(p.mu)} <span className="muted small">±{displaySd(p.sd)}</span>
                </td>
                <td className="num hide-sm">{p.games.toLocaleString()}</td>
                <td className="hide-sm muted">{p.last}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {filtered.length > limit && (
          <button className="link more" onClick={() => setLimit(limit + PAGE)}>
            Show more
          </button>
        )}
      </div>
    </section>
  );
}
