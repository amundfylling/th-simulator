/**
 * LightGBM layer (M9): pre-event win/draw/loss odds for a pair of players, turned into a
 * per-pair shift on top of the rating gap for the engine's head-to-head map.
 *
 * The model is exported by thsim.models.gbm.to_json. It is order-free: odds for (a, b)
 * average the prediction for (a, b) and the mirrored one for (b, a).
 */

export interface GbmTree {
  /** Split feature per node. */
  f: number[];
  /** Threshold per node: go left when x <= t. */
  t: number[];
  /** Children: >= 0 is a node index, < 0 is ~leafIndex. */
  l: number[];
  r: number[];
  /** Missing handling: 0 none (NaN -> 0), 1 zero (0 and NaN go the default way), 2 NaN. */
  m: number[];
  /** Default direction is left. */
  dl: number[];
  v: number[];
  root: number;
}

export interface GbmJson {
  features: string[];
  /** Per-player feature names, in the order of each player's feature vector. */
  side: string[];
  num_class: number;
  trees: GbmTree[];
}

/** Everything the model needs about one player (side features in GbmJson.side order). */
export interface PairPlayer {
  id: number;
  mu: number;
  sd: number;
  feat: (number | null)[];
}

export interface PairContext {
  /** Scorpion level (1 = majors .. 6), -1 when unknown. */
  level: number;
  bordshockey: number;
}

function leafValue(tree: GbmTree, x: Float64Array): number {
  let k = tree.root;
  while (k >= 0) {
    let v = x[tree.f[k]];
    const miss = tree.m[k];
    let left: boolean;
    if (miss === 2 && Number.isNaN(v)) left = tree.dl[k] === 1;
    else if (miss === 1 && (Number.isNaN(v) || Math.abs(v) <= 1e-35)) left = tree.dl[k] === 1;
    else {
      if (Number.isNaN(v)) v = 0;
      left = v <= tree.t[k];
    }
    k = left ? tree.l[k] : tree.r[k];
  }
  return tree.v[~k];
}

export class Gbm {
  private readonly col: Map<string, number>;
  private readonly x: Float64Array;

  constructor(readonly json: GbmJson) {
    this.col = new Map(json.features.map((f, i) => [f, i]));
    this.x = new Float64Array(json.features.length);
  }

  /** Class probabilities for one feature vector (in json.features order). */
  predictRow(x: Float64Array): [number, number, number] {
    const raw = [0, 0, 0];
    this.json.trees.forEach((t, i) => (raw[i % 3] += leafValue(t, x)));
    const m = Math.max(...raw);
    const e = raw.map((r) => Math.exp(r - m));
    const s = e[0] + e[1] + e[2];
    return [e[0] / s, e[1] / s, e[2] / s];
  }

  private fill(a: PairPlayer, b: PairPlayer, h2h: number, meet: number, ctx: PairContext): Float64Array {
    const x = this.x;
    const set = (name: string, v: number | null | undefined) => {
      const i = this.col.get(name);
      if (i !== undefined) x[i] = v === null || v === undefined ? NaN : v;
    };
    set("d", a.mu - b.mu);
    set("sd", Math.sqrt(a.sd * a.sd + b.sd * b.sd));
    set("h2h", h2h);
    set("meet", meet);
    set("po", 0);
    set("level", ctx.level);
    set("bordshockey", ctx.bordshockey);
    set("team", 0);
    this.json.side.forEach((c, j) => {
      set(`${c}_1`, a.feat[j]);
      set(`${c}_2`, b.feat[j]);
    });
    return x;
  }

  /** Order-free odds that a beats / draws / loses to b. h2h and meet are from a's side. */
  pairOdds(a: PairPlayer, b: PairPlayer, h2h: number, meet: number, ctx: PairContext): [number, number, number] {
    const p = this.predictRow(this.fill(a, b, h2h, meet, ctx));
    const q = this.predictRow(this.fill(b, a, -h2h, meet, ctx));
    return [(p[0] + q[2]) / 2, (p[1] + q[1]) / 2, (p[2] + q[0]) / 2];
  }
}

const logit = (p: number) => {
  const c = Math.min(Math.max(p, 1e-6), 1 - 1e-6);
  return Math.log(c / (1 - c));
};

/**
 * Per-pair shifts ("idA-idB" with idA < idB -> shift in idA's favour) that make the engine's
 * game odds match the model's: the gap whose regulation odds give the model's win/loss
 * balance, widened back by the flattening the engine's rating draws add, minus the rating gap.
 */
export function pairShifts(
  gbm: Gbm,
  players: PairPlayer[],
  h2h: Record<string, number | [number, number]>,
  beta: number,
  ctx: PairContext,
): Record<string, number> {
  const out: Record<string, number> = {};
  const sorted = [...players].sort((p, q) => p.id - q.id);
  for (let i = 0; i < sorted.length; i++)
    for (let j = i + 1; j < sorted.length; j++) {
      const a = sorted[i];
      const b = sorted[j];
      const hv = h2h[`${a.id}-${b.id}`];
      const [edge, meet] = hv === undefined ? [0, 0] : typeof hv === "number" ? [hv, 0] : hv;
      const [pa, , pb] = gbm.pairOdds(a, b, edge, meet, ctx);
      const inv = (logit(pa) - logit(pb)) / (2 * beta);
      const sd = Math.sqrt(a.sd * a.sd + b.sd * b.sd);
      const k = 1 / Math.sqrt(1 + (Math.PI * beta * beta * sd * sd) / 8);
      out[`${a.id}-${b.id}`] = Math.round((inv / k - (a.mu - b.mu)) * 1e4) / 1e4;
    }
  return out;
}
