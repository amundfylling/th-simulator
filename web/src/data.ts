import type { GameModelJson } from "./engine/types";

export interface RankedPlayer {
  id: number;
  name: string;
  country: string;
  ithfId: number | null;
  worldRank: number | null;
  rankingPoints: number | null;
  mu: number;
  sd: number;
  games: number;
  last: string;
}

export interface Meta {
  generated: string;
  games: number;
  latest_game: string;
  players: number;
  ranking_available: boolean;
  rating_params?: { prior_mean: number; prior_sd: number; level_prior: number };
  tournament_backtest?: {
    window: [string, string];
    variants: Record<string, { events: number; winner_ll: number; uniform_ll: number; rps: number; top3_brier: number; fav_won: number }>;
  };
}

export interface AppData {
  players: RankedPlayer[];
  model: GameModelJson;
  meta: Meta;
  /** Head-to-head edges "idA-idB" (idA < idB) -> rating shift in idA's favour; empty when not exported. */
  h2h: Record<string, number>;
}

/** The head-to-head edges between the given players only (what the engine needs for one event). */
export function h2hFor(all: Record<string, number>, ids: number[]): Record<string, number> {
  const out: Record<string, number> = {};
  const sorted = [...ids].sort((a, b) => a - b);
  for (let i = 0; i < sorted.length; i++)
    for (let j = i + 1; j < sorted.length; j++) {
      const key = `${sorted[i]}-${sorted[j]}`;
      if (key in all) out[key] = all[key];
    }
  return out;
}

export async function loadData(): Promise<AppData> {
  const base = `${import.meta.env.BASE_URL}data/`;
  const [p, model, meta, h2h] = await Promise.all([
    fetch(`${base}players.json`).then((r) => r.json()),
    fetch(`${base}model.json`).then((r) => r.json()),
    fetch(`${base}meta.json`).then((r) => r.json()),
    // Optional: older exports have no head-to-head file.
    fetch(`${base}h2h.json`).then((r) => (r.ok ? r.json() : {})).catch(() => ({})),
  ]);
  const cols: string[] = p.columns;
  const at = (k: string) => cols.indexOf(k);
  const players: RankedPlayer[] = p.rows.map((r: unknown[]) => ({
    id: r[at("id")] as number,
    name: r[at("name")] as string,
    country: (r[at("country")] as string) ?? "",
    ithfId: r[at("ithf_id")] as number | null,
    worldRank: r[at("world_rank")] as number | null,
    rankingPoints: r[at("ranking_points")] as number | null,
    mu: r[at("mu")] as number,
    sd: r[at("sd")] as number,
    games: r[at("games")] as number,
    last: r[at("last")] as string,
  }));
  return { players, model, meta, h2h: h2h ?? {} };
}

/** Display rating: 1500 + 100 per goal of expected margin. */
export const displayRating = (mu: number) => Math.round(1500 + 100 * mu);
export const displaySd = (sd: number) => Math.round(100 * sd);
