import { displayRating, type RankedPlayer } from "../data";

interface Props {
  entries: RankedPlayer[];
  onChange: (ids: number[]) => void;
}

export function EntryList({ entries, onChange }: Props) {
  const ids = entries.map((p) => p.id);
  const move = (i: number, d: number) => {
    const j = i + d;
    if (j < 0 || j >= ids.length) return;
    const next = ids.slice();
    [next[i], next[j]] = [next[j], next[i]];
    onChange(next);
  };
  const sortBy = (key: "rank" | "rating") => {
    const sorted = entries.slice().sort((a, b) =>
      key === "rating" ? b.mu - a.mu : (a.worldRank ?? 1e9) - (b.worldRank ?? 1e9) || b.mu - a.mu,
    );
    onChange(sorted.map((p) => p.id));
  };
  return (
    <section className="card entries">
      <header className="card-head">
        <h2>
          <span className="step">2</span> Entry list <span className="badge">{entries.length}</span>
        </h2>
        <p className="muted">The order is the seeding. Seed 1 is on top.</p>
      </header>
      {entries.length === 0 ? (
        <p className="empty">No players yet. Add some from the list.</p>
      ) : (
        <>
          <div className="quick">
            Seed by
            <button className="chip" onClick={() => sortBy("rank")}>
              world rank
            </button>
            <button className="chip" onClick={() => sortBy("rating")}>
              rating
            </button>
            <button className="chip ghost" onClick={() => onChange([])}>
              Clear
            </button>
          </div>
          <ol className="entry-list">
            {entries.map((p, i) => (
              <li key={p.id}>
                <span className="seed">{i + 1}</span>
                <span className="name">{p.name}</span>
                <span className="muted small">{displayRating(p.mu)}</span>
                <span className="row-actions">
                  <button aria-label={`Move ${p.name} up`} onClick={() => move(i, -1)} disabled={i === 0}>
                    ↑
                  </button>
                  <button aria-label={`Move ${p.name} down`} onClick={() => move(i, 1)} disabled={i === entries.length - 1}>
                    ↓
                  </button>
                  <button aria-label={`Remove ${p.name}`} onClick={() => onChange(ids.filter((x) => x !== p.id))}>
                    ✕
                  </button>
                </span>
              </li>
            ))}
          </ol>
        </>
      )}
    </section>
  );
}
