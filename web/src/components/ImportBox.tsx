import { useState } from "react";
import type { RankedPlayer } from "../data";
import { convertTournament, fetchTournament, tournamentIdFromLink, type ImportResult, type RatingDefaults } from "../import/tableHockeyEu";

interface Props {
  players: RankedPlayer[];
  defaults: RatingDefaults;
  imported: { link: string; result: ImportResult } | null;
  onImport: (link: string, result: ImportResult) => void;
  onClear: () => void;
}

/**
 * table-hockey.eu does not allow other sites to read its API, so requests go through a relay.
 * VITE_TH_PROXY (a URL prefix, set at build time) is tried first, then public CORS relays.
 */
const RELAYS: ((url: string) => string)[] = [
  ...(import.meta.env.VITE_TH_PROXY ? [(u: string) => `${import.meta.env.VITE_TH_PROXY}${encodeURIComponent(u)}`] : []),
  (u) => `https://api.allorigins.win/raw?url=${encodeURIComponent(u)}`,
  (u) => `https://api.codetabs.com/v1/proxy/?quest=${encodeURIComponent(u)}`,
];

async function load(id: number) {
  try {
    return await fetchTournament(id);
  } catch (e) {
    let last = e;
    for (const relay of RELAYS) {
      try {
        return await fetchTournament(id, (url, init) => fetch(relay(String(url)), init));
      } catch (err) {
        last = err;
      }
    }
    throw last;
  }
}

export async function importLink(link: string, players: RankedPlayer[], defaults: RatingDefaults): Promise<ImportResult> {
  const id = tournamentIdFromLink(link);
  if (id === null) throw new Error("Paste a link like https://table-hockey.eu/tournaments/33");
  let data;
  try {
    data = await load(id);
  } catch (e) {
    if (e instanceof TypeError)
      throw new Error("Your browser could not read table-hockey.eu from this site (the request was blocked or the site is down).");
    throw e;
  }
  return convertTournament(data.tournament, data.stages, players, defaults);
}

export function ImportBox({ players, defaults, imported, onImport, onClear }: Props) {
  const [link, setLink] = useState(imported?.link ?? "");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const go = async (l = link) => {
    setBusy(true);
    setErr(null);
    try {
      onImport(l, await importLink(l, players, defaults));
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const r = imported?.result;
  return (
    <section className="card import">
      <header className="card-head">
        <h2>Load a tournament from table-hockey.eu</h2>
        <p className="muted">Paste a tournament link to get its players, its real format and every game played so far.</p>
      </header>
      <form
        className="import-row"
        onSubmit={(e) => {
          e.preventDefault();
          go();
        }}
      >
        <input
          type="url"
          inputMode="url"
          placeholder="https://table-hockey.eu/tournaments/33"
          value={link}
          onChange={(e) => setLink(e.target.value)}
          aria-label="table-hockey.eu tournament link"
        />
        <button className="primary" disabled={busy || !link.trim()}>
          {busy ? "Loading…" : "Load"}
        </button>
      </form>
      {err && <p className="error small">{err}</p>}
      {r && imported && (
        <div className="import-summary">
          <p>
            <strong>{r.name}</strong>
            {r.status ? <span className="badge">{r.status}</span> : null} · {r.entries.length} players · {r.stages.length} stage
            {r.stages.length === 1 ? "" : "s"} · {r.played.length} game{r.played.length === 1 ? "" : "s"} already played
          </p>
          {r.notes.length > 0 && (
            <ul className="muted small">
              {r.notes.map((n) => (
                <li key={n}>{n}</li>
              ))}
            </ul>
          )}
          <div className="quick">
            <button className="chip" disabled={busy} onClick={() => go(imported.link)}>
              Refresh results
            </button>
            <button className="chip ghost" onClick={onClear}>
              Clear
            </button>
          </div>
        </div>
      )}
    </section>
  );
}
