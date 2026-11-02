/** Single-game simulator built from the exported game model. */
import type { GameModelJson } from "./types";
import type { Rng } from "./rng";

interface Alias {
  prob: Float64Array;
  alias: Int32Array;
  fav: Int8Array;
  dog: Int8Array;
}

function buildAlias(entries: [number, number, number][]): Alias {
  const n = entries.length;
  const total = entries.reduce((s, e) => s + e[2], 0);
  const scaled = entries.map((e) => (e[2] / total) * n);
  const prob = new Float64Array(n);
  const alias = new Int32Array(n);
  const small: number[] = [];
  const large: number[] = [];
  scaled.forEach((p, i) => (p < 1 ? small : large).push(i));
  while (small.length && large.length) {
    const s = small.pop()!;
    const l = large.pop()!;
    prob[s] = scaled[s];
    alias[s] = l;
    scaled[l] = scaled[l] + scaled[s] - 1;
    (scaled[l] < 1 ? small : large).push(l);
  }
  for (const i of [...small, ...large]) {
    prob[i] = 1;
    alias[i] = i;
  }
  return {
    prob,
    alias,
    fav: Int8Array.from(entries.map((e) => e[0])),
    dog: Int8Array.from(entries.map((e) => e[1])),
  };
}

export interface GameOut {
  g1: number;
  g2: number;
  ot: boolean;
}

export class GameSampler {
  private readonly tables: Alias[][][]; // [kind 0 rr / 1 po][outcome 0 fav / 1 draw / 2 dog][bin]
  private readonly xBin: number;
  private readonly nBins: number;
  private readonly betaX: number;

  constructor(readonly model: GameModelJson) {
    const s = model.scoreline;
    this.xBin = s.x_bin;
    this.nBins = s.n_bins;
    this.betaX = s.beta;
    this.tables = (["rr", "po"] as const).map((kind) =>
      (["fav", "draw", "dog"] as const).map((k) => {
        const bins = s.tables[kind][k];
        // A bin without data borrows the nearest bin that has some.
        return bins.map((_, b) => {
          for (let off = 0; off < bins.length; off++) {
            for (const j of [b - off, b + off]) if (j >= 0 && j < bins.length && bins[j].length) return buildAlias(bins[j]);
          }
          throw new Error(`Empty scoreline table ${kind}/${k}`);
        });
      }),
    );
  }

  /** Regulation outcome probabilities [p1 wins, draw, p2 wins] for gap d. */
  regProbs(d: number, playoff: boolean): [number, number, number] {
    const { beta, c } = playoff ? this.model.regulation.po : this.model.regulation.rr;
    const pa = sigmoid(beta * d - c);
    const pb = sigmoid(-beta * d - c);
    return [pa, Math.max(0, 1 - pa - pb), pb];
  }

  /**
   * Play one game with gap d = strength(p1) - strength(p2).
   * needWinner: a drawn regulation score goes to golden-goal overtime.
   */
  play(d: number, playoff: boolean, needWinner: boolean, rng: Rng, out: GameOut): GameOut {
    const [pa, pd] = this.regProbs(d, playoff);
    const u = rng.next();
    const y = u < pa ? 0 : u < pa + pd ? 1 : 2; // from p1's view
    const p1Fav = d >= 0;
    const fy = p1Fav ? y : 2 - y;
    let bin = Math.floor((this.betaX * Math.abs(d)) / this.xBin);
    if (bin >= this.nBins) bin = this.nBins - 1;
    const t = this.tables[playoff ? 1 : 0][fy][bin];
    let k = rng.int(t.prob.length);
    if (rng.next() >= t.prob[k]) k = t.alias[k];
    const fav = t.fav[k];
    const dog = t.dog[k];
    out.g1 = p1Fav ? fav : dog;
    out.g2 = p1Fav ? dog : fav;
    out.ot = false;
    if (needWinner && out.g1 === out.g2) {
      out.ot = true;
      if (rng.next() < sigmoid(this.model.overtime.beta * d)) out.g1 += 1;
      else out.g2 += 1;
    }
    return out;
  }
}

export function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}
