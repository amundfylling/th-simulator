/// <reference lib="webworker" />
import { Tournament } from "./engine/simulate";
import type { GameModelJson, PlayedGame, Player, StageConfig } from "./engine/types";

export interface SimRequest {
  players: Player[];
  stages: StageConfig[];
  played?: PlayedGame[];
  model: GameModelJson;
  runs: number;
  seed: number;
  ratingUncertainty: boolean;
}

self.onmessage = (e: MessageEvent<SimRequest>) => {
  const req = e.data;
  try {
    const t = new Tournament(req.players, req.stages, req.model, req.played ?? []);
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
