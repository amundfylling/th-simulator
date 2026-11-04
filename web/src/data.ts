import type { GbmJson, PairContext, PairPlayer } from "./engine/gbm";
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
  /** Frozen features for the LightGBM layer (GbmJson.side order); absent in older exports. */
  feat?: (number | null)[];
}

export interface Meta {
  generated: string;
  games: number;
  latest_game: string;
  players: number;
  ranking_available: boolean;
  rating_params?: { prior_mean: number; prior_sd: number; level_prior: number };
  /** LightGBM layer: feature vector of a player with no games (mu is replaced per player). */
  gbm?: { rounds: number; newcomer: (number | null)[]; kal_level_prior?: number } | null;
  tournament_backtest?: {
    window: [string, string];
    variants: Record<string, { events: number; winner_ll: number; uniform_ll: number; rps: number; top3_brier: number; fav_won: number }>;
  };
}

export interface AppData {
  players: RankedPlayer[];
  model: GameModelJson;
  meta: Meta;
  /** Head-to-head "idA-idB" (idA < idB) -> rating shift in idA's favour, or [shift, meetings]; empty when not exported. */
  h2h: Record<string, H2hValue>;
  /** LightGBM layer (gbm.json); null when not exported. */
  gbm: GbmJson | null;
}

export type H2hValue = number | [number, number];

/** The head-to-head entries between the given players only (what the engine needs for one event). */
export function h2hFor(all: Record<string, H2hValue>, ids: number[]): Record<string, H2hValue> {
  const out: Record<string, H2hValue> = {};
  const sorted = [...ids].sort((a, b) => a - b);
  for (let i = 0; i < sorted.length; i++)
    for (let j = i + 1; j < sorted.length; j++) {
      const key = `${sorted[i]}-${sorted[j]}`;
      if (key in all) out[key] = all[key];
    }
  return out;
}

/**
 * What the worker needs to run the LightGBM layer for these entrants, or null without a model.
 * Players without exported features (newcomers from an import) use the newcomer vector with
 * their own rating.
 */
export function gbmRequest(
  data: AppData,
  entries: RankedPlayer[],
  level: number,
): { json: GbmJson; players: PairPlayer[]; ctx: PairContext } | null {
  const newcomer = data.meta.gbm?.newcomer;
  if (!data.gbm || !newcomer) return null;
  const muAt = data.gbm.side.indexOf("mu");
  const kalAt = data.gbm.side.indexOf("kal");
  // The newcomer vector is level-neutral; the Kalman prior moves with the event level like the composite rating does.
  const kalShift = level > 0 ? -(data.meta.gbm?.kal_level_prior ?? 0) * (level - 3) : 0;
  const players = entries.map((p) => {
    let feat = p.feat;
    if (!feat) {
      feat = [...newcomer];
      if (muAt >= 0) feat[muAt] = p.mu;
      if (kalAt >= 0 && feat[kalAt] !== null) feat[kalAt] = (feat[kalAt] as number) + kalShift;
    }
    return { id: p.id, mu: p.mu, sd: p.sd, feat };
  });
  return { json: data.gbm, players, ctx: { level: level > 0 ? level : -1, bordshockey: 0 } };
}

export async function loadData(): Promise<AppData> {
  const base = `${import.meta.env.BASE_URL}data/`;
  const [p, model, meta, h2h, gbm] = await Promise.all([
    fetch(`${base}players.json`).then((r) => r.json()),
    fetch(`${base}model.json`).then((r) => r.json()),
    fetch(`${base}meta.json`).then((r) => r.json()),
    // Optional: older exports have no head-to-head file.
    fetch(`${base}h2h.json`).then((r) => (r.ok ? r.json() : {})).catch(() => ({})),
    fetch(`${base}gbm.json`).then((r) => (r.ok ? r.json() : null)).catch(() => null),
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
    feat: at("feat") >= 0 ? (r[at("feat")] as (number | null)[]) : undefined,
  }));
  return { players, model, meta, h2h: h2h ?? {}, gbm };
}

/** Display rating: 1500 + 100 per goal of expected margin. */
export const displayRating = (mu: number) => Math.round(1500 + 100 * mu);
export const displaySd = (sd: number) => Math.round(100 * sd);
