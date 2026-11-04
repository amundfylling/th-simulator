/**
 * Simulate reconstructed historical tournaments with the website's engine.
 * Usage: tsx scripts/backtest.ts events.json model.json out.json runs variant...
 * Variants: kalman | kalman_form<sd> | kalman_nounc | elo
 * With GBM_MODEL=path/to/gbm.json and events whose players carry `feat`, the pair shifts come
 * from the LightGBM layer (the same code the website runs) instead of the plain h2h edges.
 */
import { readFileSync, writeFileSync } from "node:fs";
import { Gbm, pairShifts, type GbmJson, type PairContext } from "../src/engine/gbm";
import { Tournament } from "../src/engine/simulate";
import type { GameModelJson, Player, StageConfig } from "../src/engine/types";

interface EventPlayer {
  id: number;
  name: string;
  mu: number;
  sd: number;
  elo: number;
  /** Frozen pre-event features for the LightGBM layer (GbmJson.side order). */
  feat?: (number | null)[];
}
interface Event {
  tournament_uid: number;
  stages: StageConfig[];
  players: EventPlayer[];
  /** Optional head-to-head edges ("idA-idB" -> shift for idA), frozen at the event's start. */
  h2h?: Record<string, number | [number, number]>;
  ctx?: PairContext;
}

const [eventsPath, modelPath, outPath, runsArg, ...variants] = process.argv.slice(2);
const data = JSON.parse(readFileSync(eventsPath, "utf8"));
const model: GameModelJson = JSON.parse(readFileSync(modelPath, "utf8"));
const runs = Number(runsArg);
const gbm = process.env.GBM_MODEL ? new Gbm(JSON.parse(readFileSync(process.env.GBM_MODEL, "utf8")) as GbmJson) : null;
const shifts = (ev: Event): Record<string, number> => {
  const h2h = ev.h2h ?? {};
  if (gbm && ev.players.every((p) => p.feat)) {
    const pp = ev.players.map((p) => ({ id: p.id, mu: p.mu, sd: p.sd, feat: p.feat! }));
    return pairShifts(gbm, pp, h2h, model.regulation.rr.beta, ev.ctx ?? { level: -1, bordshockey: 0 });
  }
  return Object.fromEntries(Object.entries(h2h).map(([k, v]) => [k, typeof v === "number" ? v : v[0]]));
};
const out: Record<string, Record<string, { p: number[][] }>> = {};
const t0 = Date.now();

for (const variant of variants) {
  out[variant] = {};
  const form = variant.startsWith("kalman_form") ? Number(variant.slice("kalman_form".length)) : 0;
  data.events.forEach((ev: Event, k: number) => {
    const players: Player[] = ev.players.map((p) =>
      variant === "elo"
        ? { id: p.id, name: p.name, mu: (p.elo - 1500) * data.elo_scale, sd: 0 }
        : { id: p.id, name: p.name, mu: p.mu, sd: p.sd },
    );
    // Entry order = seeding by rating (matters for knockout-only events and final tiebreaks).
    const order = players.map((_, i) => i).sort((a, b) => players[b].mu - players[a].mu);
    const t = new Tournament(order.map((i) => players[i]), ev.stages, model, [], variant === "elo" ? {} : shifts(ev));
    const res = t.simulate({ runs, seed: 1000 + k, ratingUncertainty: variant !== "kalman_nounc" && variant !== "elo", formSd: form });
    const p: number[][] = new Array(players.length);
    order.forEach((orig, j) => {
      p[orig] = res.placeCounts[j].map((c) => Math.round((c / runs) * 1e5) / 1e5);
    });
    out[variant][ev.tournament_uid] = { p };
  });
  console.error(`${variant}: ${data.events.length} events, ${((Date.now() - t0) / 1000).toFixed(0)} s`);
}
writeFileSync(outPath, JSON.stringify({ runs, variants: out }));
