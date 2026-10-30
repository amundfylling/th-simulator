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
}

export async function loadData(): Promise<AppData> {
  const base = `${import.meta.env.BASE_URL}data/`;
  const [p, model, meta] = await Promise.all([
    fetch(`${base}players.json`).then((r) => r.json()),
    fetch(`${base}model.json`).then((r) => r.json()),
    fetch(`${base}meta.json`).then((r) => r.json()),
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
  return { players, model, meta };
}

/** Display rating: 1500 + 100 per goal of expected margin. */
export const displayRating = (mu: number) => Math.round(1500 + 100 * mu);
export const displaySd = (sd: number) => Math.round(100 * sd);
