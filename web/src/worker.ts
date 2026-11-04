/// <reference lib="webworker" />
import { Gbm, pairShifts, type GbmJson, type PairContext, type PairPlayer } from "./engine/gbm";
import { Tournament } from "./engine/simulate";
import type { GameModelJson, PlayedGame, Player, StageConfig } from "./engine/types";

export interface SimRequest {
  players: Player[];
  stages: StageConfig[];
  played?: PlayedGame[];
  h2h?: Record<string, number | [number, number]>;
  /** LightGBM layer: when set, per-pair shifts come from the model (players need features). */
  gbm?: { json: GbmJson; players: PairPlayer[]; ctx: PairContext } | null;
  model: GameModelJson;
  runs: number;
  seed: number;
  ratingUncertainty: boolean;
}

self.onmessage = (e: MessageEvent<SimRequest>) => {
  const req = e.data;
  try {
    const h2h = req.h2h ?? {};
    const shifts = req.gbm
      ? pairShifts(new Gbm(req.gbm.json), req.gbm.players, h2h, req.model.regulation.rr.beta, req.gbm.ctx)
      : Object.fromEntries(Object.entries(h2h).map(([k, v]) => [k, typeof v === "number" ? v : v[0]]));
    const t = new Tournament(req.players, req.stages, req.model, req.played ?? [], shifts);
    const result = t.simulate({
      runs: req.runs,
      seed: req.seed,
      ratingUncertainty: req.ratingUncertainty,
      onProgress: (done) => self.postMessage({ type: "progress", done }),
    });
    self.postMessage({ type: "done", result });
  } catch (err) {
    self.postMessage({ type: "error", message: (err as Error).message });
  }
};
