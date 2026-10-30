/**
 * Monte Carlo tournament engine.
 *
 * A tournament is a list of stages (groups, Swiss, knockout). Each stage takes its
 * entrants from the entry list or from slices of earlier stages' rankings, and returns
 * a full ranking of its own entrants. Final places 1..N come from the stages that set
 * `places`, latest stage first, so every place is defined.
 *
 * Every run draws each player's strength once (posterior draw + optional form shock)
 * and keeps it for the whole run.
 */
import { GameSampler, type GameOut } from "./gameModel";
import { Rng } from "./rng";
import type {
  GameModelJson,
  GroupStageConfig,
  KnockoutStageConfig,
  PlayedGame,
  Player,
  SimOptions,
  SimResult,
  StageConfig,
  StageStats,
  SwissStageConfig,
  Tiebreak,
} from "./types";

const DEFAULT_POINTS: [number, number, number] = [2, 1, 0];
const DEFAULT_TIEBREAK: Tiebreak[] = ["points", "h2h", "gd", "gf", "lot"];

interface Standing {
  p: number; // player index
  seed: number; // position in the stage's entry order (lower = better)
  points: number;
  gf: number;
  ga: number;
  wins: number;
  games: number;
  h2h: number;
  lot: number;
}

/** Games stored per run for carry-over: key "stage|a|b" with a < b, goals from a's view. */
type GameLog = Map<string, [number, number][]>;

export class Tournament {
  readonly n: number;
  private readonly index = new Map<number, number>();
  private readonly played = new Map<string, PlayedGame[]>();
  private readonly sampler: GameSampler;
  private readonly strength: Float64Array;
  private readonly out: GameOut = { g1: 0, g2: 0, ot: false };
  private readonly carrySources: Set<string>;

  constructor(
    readonly players: Player[],
    readonly stages: StageConfig[],
    model: GameModelJson,
    played: PlayedGame[] = [],
  ) {
    this.n = players.length;
    players.forEach((p, i) => this.index.set(p.id, i));
    this.sampler = new GameSampler(model);
    this.strength = new Float64Array(this.n);
    for (const g of played) {
      const key = pairKey(g.stage, g.p1, g.p2);
      if (!this.played.has(key)) this.played.set(key, []);
      this.played.get(key)!.push(g);
    }
    validate(stages);
    this.carrySources = new Set(stages.flatMap((s) => (s.type === "groups" && s.carryFrom ? [s.carryFrom] : [])));
  }

  simulate(opts: SimOptions): SimResult {
    const t0 = Date.now();
    const rng = new Rng(opts.seed ?? 12345);
    const n = this.n;
    const placeCounts = Array.from({ length: n }, () => new Array<number>(n).fill(0));
    const stats = this.stages.map((s) => initStats(s, n));
    const finals: Record<string, number> = {};
    const unc = opts.ratingUncertainty ?? true;
    const formSd = opts.formSd ?? this.sampler.model.form_sd ?? 0;
    for (let run = 0; run < opts.runs; run++) {
      for (let i = 0; i < n; i++) {
        const p = this.players[i];
        this.strength[i] = p.mu + (unc ? p.sd * rng.normal() : 0) + (formSd ? formSd * rng.normal() : 0);
      }
      const places = this.runOnce(rng, stats, finals);
      for (let i = 0; i < n; i++) placeCounts[i][places[i]]++;
      if (opts.onProgress && (run + 1) % 500 === 0) opts.onProgress(run + 1);
    }
    return {
      runs: opts.runs,
      players: this.players,
      placeCounts,
      stages: stats,
      finals,
      elapsedMs: Date.now() - t0,
    };
  }

  /** One full tournament. Returns the 0-based final place of every player. */
  runOnce(rng: Rng, stats: StageStats[], finals: Record<string, number>): Int32Array {
    const rankings = new Map<string, number[]>();
    rankings.set("entry", this.players.map((_, i) => i));
    const log: GameLog = new Map();
    this.stages.forEach((stage, si) => {
      const entrants: number[] = [];
      for (const src of stage.source) {
        const r = rankings.get(src.from)!;
        const [a, b] = src.ranks ?? [1, r.length];
        for (let k = a - 1; k < Math.min(b, r.length); k++) entrants.push(r[k]);
      }
      const st = stats[si];
      for (const p of entrants) st.entered[p]++;
      let ranking: number[];
      if (stage.type === "groups") ranking = this.groups(stage, entrants, rng, st, log);
      else if (stage.type === "swiss") ranking = this.swiss(stage, entrants, rng, st);
      else ranking = this.knockout(stage, entrants, rng, st, finals);
      rankings.set(stage.id, ranking);
    });
    return this.finalPlaces(rankings);
  }

  private finalPlaces(rankings: Map<string, number[]>): Int32Array {
    const n = this.n;
    const place = new Int32Array(n).fill(-1);
    const taken = new Uint8Array(n);
    const assign = (p: number, want: number) => {
      let k = Math.max(0, Math.min(want, n - 1));
      while (k < n && taken[k]) k++;
      if (k >= n) {
        k = 0;
        while (taken[k]) k++;
      }
      taken[k] = 1;
      place[p] = k;
    };
    for (let si = this.stages.length - 1; si >= 0; si--) {
      const stage = this.stages[si];
      if (stage.places === undefined) continue;
      const r = rankings.get(stage.id)!;
      r.forEach((p, k) => {
        if (place[p] < 0) assign(p, stage.places! - 1 + k);
      });
    }
    // Anyone left: by the latest stage they played, in that stage's order.
    for (let si = this.stages.length - 1; si >= -1; si--) {
      const r = rankings.get(si < 0 ? "entry" : this.stages[si].id)!;
      for (const p of r) if (place[p] < 0) assign(p, 0);
    }
    return place;
  }

  private game(stageId: string, a: number, b: number, playoff: boolean, needWinner: boolean, rng: Rng, nth: number): GameOut {
    const fixed = this.played.size ? this.played.get(pairKey(stageId, this.players[a].id, this.players[b].id)) : undefined;
    const out = this.out;
    if (fixed && nth < fixed.length) {
      const g = fixed[nth];
      const aIsP1 = g.p1 === this.players[a].id;
      out.g1 = aIsP1 ? g.g1 : g.g2;
      out.g2 = aIsP1 ? g.g2 : g.g1;
      out.ot = !!g.ot;
      return out;
    }
    return this.sampler.play(this.strength[a] - this.strength[b], playoff, needWinner, rng, out);
  }

  // --- round-robin groups ------------------------------------------------------

  private groups(stage: GroupStageConfig, entrants: number[], rng: Rng, st: StageStats, log: GameLog): number[] {
    const G = Math.max(1, Math.min(stage.groups, entrants.length));
    const legs = stage.legs ?? 1;
    const pts = stage.points ?? DEFAULT_POINTS;
    const tiebreak = stage.tiebreak ?? DEFAULT_TIEBREAK;
    let groups: number[][];
    if (stage.groupsExplicit) {
      const inStage = new Set(entrants);
      groups = stage.groupsExplicit.map((ids) => ids.map((id) => this.index.get(id)!).filter((p) => p !== undefined && inStage.has(p)));
    } else {
      const order = stage.seeding === "random" ? rng.shuffle(entrants.slice()) : entrants;
      groups = assignGroups(order, G, stage.seeding === "block" ? "block" : "snake");
    }
    const seedOf = new Map(entrants.map((p, i) => [p, i]));
    const byPosition: Standing[][] = [];
    for (const members of groups) {
      const table = members.map((p) => standing(p, seedOf.get(p)!, rng));
      const m = members.length;
      const local = new Map(members.map((p, x) => [p, x]));
      const res: ([number, number][] | undefined)[] = new Array(m * m);
      const keepLog = this.carrySources.has(stage.id);
      for (let x = 0; x < members.length; x++) {
        for (let y = x + 1; y < members.length; y++) {
          const a = members[x];
          const b = members[y];
          let games: [number, number][] | undefined;
          if (stage.carryFrom) games = log.get(logKey(stage.carryFrom, a, b));
          if (!games) {
            games = [];
            for (let l = 0; l < legs; l++) {
              const o = this.game(stage.id, a, b, false, false, rng, l);
              games.push([o.g1, o.g2]);
            }
          }
          res[x * m + y] = games;
          if (keepLog) log.set(logKey(stage.id, a, b), a < b ? games : games.map(([u, v]) => [v, u]));
          for (const [ga, gb] of games) addGame(table[x], table[y], ga, gb, pts);
        }
      }
      rankTable(table, tiebreak, (a, b) => {
        const x = local.get(a)!;
        const y = local.get(b)!;
        return x < y ? res[x * m + y] : res[y * m + x]?.map(([u, v]) => [v, u] as [number, number]);
      }, pts);
      table.forEach((s, k) => {
        (byPosition[k] ??= []).push(s);
        st.position![s.p][k]++;
        st.pointsSum![s.p] += s.points;
        st.gamesSum![s.p] += s.games;
      });
    }
    // Overall ranking: by group position, then per-game record across groups.
    const ranking: number[] = [];
    for (const tier of byPosition) {
      tier.sort(
        (u, v) =>
          v.points / Math.max(1, v.games) - u.points / Math.max(1, u.games) ||
          (v.gf - v.ga) / Math.max(1, v.games) - (u.gf - u.ga) / Math.max(1, u.games) ||
          v.gf / Math.max(1, v.games) - u.gf / Math.max(1, u.games) ||
          u.seed - v.seed,
      );
      for (const s of tier) ranking.push(s.p);
    }
    return ranking;
  }

  // --- Swiss ----------------------------------------------------------------------

  private swiss(stage: SwissStageConfig, entrants: number[], rng: Rng, st: StageStats): number[] {
    const pts = stage.points ?? DEFAULT_POINTS;
    const table = entrants.map((p, i) => standing(p, i, rng));
    const met = new Set<string>();
    const results = new Map<string, [number, number][]>(); // key min|max, goals from min's view
    const byeHad = new Set<number>();
    for (let round = 0; round < stage.rounds; round++) {
      const order = table.slice().sort((u, v) => v.points - u.points || (v.gf - v.ga) - (u.gf - u.ga) || u.seed - v.seed);
      if (order.length % 2 === 1) {
        // lowest-ranked player without a bye sits out and scores a win
        let k = order.length - 1;
        while (k > 0 && byeHad.has(order[k].p)) k--;
        const bye = order.splice(k, 1)[0];
        byeHad.add(bye.p);
        bye.points += pts[0];
      }
      const free = order.slice();
      while (free.length > 1) {
        const a = free.shift()!;
        let j = free.findIndex((b) => !met.has(logKey("", a.p, b.p)));
        if (j < 0) j = 0;
        const b = free.splice(j, 1)[0];
        met.add(logKey("", a.p, b.p));
        const o = this.game(stage.id, a.p, b.p, false, false, rng, 0);
        addGame(a, b, o.g1, o.g2, pts);
        const key = logKey("", a.p, b.p);
        const prev = results.get(key) ?? [];
        prev.push(a.p < b.p ? [o.g1, o.g2] : [o.g2, o.g1]);
        results.set(key, prev);
      }
    }
    rankTable(table, stage.tiebreak ?? ["points", "gd", "gf", "lot"], (a, b) => {
      const g = results.get(logKey("", a, b));
      return a < b ? g : g?.map(([u, v]) => [v, u] as [number, number]);
    }, pts);
    for (const s of table) {
      st.pointsSum![s.p] += s.points;
      st.gamesSum![s.p] += s.games;
    }
    return table.map((s) => s.p);
  }

  // --- knockout -------------------------------------------------------------------

  private knockout(stage: KnockoutStageConfig, entrants: number[], rng: Rng, st: StageStats, finals: Record<string, number>): number[] {
    const m = entrants.length;
    if (m === 0) return [];
    if (m === 1) {
      st.reached![entrants[0]][st.reached![entrants[0]].length - 1]++;
      return entrants.slice();
    }
    const size = nextPow2(m);
    const R = Math.log2(size);
    // reached[] is aligned to the end (last = winner) so round names match any bracket size.
    const off = st.reached![0].length - 1 - R;
    const wins = roundWins(stage.winsNeeded, R);
    const slots = bracketOrder(size).map((seed) => (seed <= m ? entrants[seed - 1] : -1));
    const seedOf = new Map(entrants.map((p, i) => [p, i]));
    const eliminated: number[][] = Array.from({ length: R }, () => []);
    let alive = slots;
    let semiLosers: number[] = [];
    for (let r = 0; r < R; r++) {
      const next: number[] = [];
      for (let k = 0; k < alive.length; k += 2) {
        const a = alive[k];
        const b = alive[k + 1];
        if (a >= 0) st.reached![a][off + r]++;
        if (b >= 0) st.reached![b][off + r]++;
        if (a < 0 || b < 0) {
          next.push(a < 0 ? b : a);
          continue;
        }
        const [hi, lo] = seedOf.get(a)! <= seedOf.get(b)! ? [a, b] : [b, a];
        if (r === R - 1) {
          const key = this.players[a].id < this.players[b].id ? `${this.players[a].id}-${this.players[b].id}` : `${this.players[b].id}-${this.players[a].id}`;
          finals[key] = (finals[key] ?? 0) + 1;
        }
        const w = this.series(stage, hi, lo, wins[r], rng);
        next.push(w);
        const loser = w === hi ? lo : hi;
        eliminated[r].push(loser);
        if (r === R - 2) semiLosers.push(loser);
      }
      alive = next;
    }
    const champion = alive[0];
    st.reached![champion][off + R]++;
    const ranking = [champion, ...eliminated[R - 1]];
    if (stage.bronze && semiLosers.length === 2) {
      semiLosers.sort((u, v) => seedOf.get(u)! - seedOf.get(v)!);
      const w = this.series({ ...stage, id: stage.id + ":bronze" }, semiLosers[0], semiLosers[1], stage.bronzeWins ?? wins[R - 1], rng);
      ranking.push(w, w === semiLosers[0] ? semiLosers[1] : semiLosers[0]);
      semiLosers = [];
    }
    for (let r = R - 2; r >= 0; r--) {
      const losers = eliminated[r].slice().sort((u, v) => seedOf.get(u)! - seedOf.get(v)!);
      for (const p of losers) if (!ranking.includes(p)) ranking.push(p);
    }
    return ranking;
  }

  /** First to `need` wins; returns the winner. Higher seed is player 1. */
  private series(stage: KnockoutStageConfig, a: number, b: number, need: number, rng: Rng): number {
    let wa = 0;
    let wb = 0;
    let nth = 0;
    const cap = 4 * need + 3; // drawn-games series: force a winner eventually
    while (wa < need && wb < need) {
      const needWinner = !stage.drawsAllowed || nth >= cap;
      const o = this.game(stage.id, a, b, true, needWinner, rng, nth++);
      if (o.g1 > o.g2) wa++;
      else if (o.g2 > o.g1) wb++;
    }
    return wa >= need ? a : b;
  }
}

// --- helpers ----------------------------------------------------------------------

function standing(p: number, seed: number, rng: Rng): Standing {
  return { p, seed, points: 0, gf: 0, ga: 0, wins: 0, games: 0, h2h: 0, lot: rng.next() };
}

function addGame(a: Standing, b: Standing, ga: number, gb: number, pts: [number, number, number]) {
  a.gf += ga;
  a.ga += gb;
  b.gf += gb;
  b.ga += ga;
  a.games++;
  b.games++;
  if (ga > gb) {
    a.points += pts[0];
    b.points += pts[2];
    a.wins++;
  } else if (gb > ga) {
    b.points += pts[0];
    a.points += pts[2];
    b.wins++;
  } else {
    a.points += pts[1];
    b.points += pts[1];
  }
}

/** Sort a table in place by the tiebreak chain. h2h = points among players tied on points. */
export function rankTable(
  table: Standing[],
  chain: Tiebreak[],
  /** Games between two players, goals from the first player's view. */
  h2hGames: (a: number, b: number) => [number, number][] | undefined,
  pts: [number, number, number],
) {
  if (chain.includes("h2h")) {
    const byPoints = new Map<number, Standing[]>();
    for (const s of table) {
      s.h2h = 0;
      const list = byPoints.get(s.points) ?? [];
      list.push(s);
      byPoints.set(s.points, list);
    }
    for (const tied of byPoints.values()) {
      if (tied.length < 2) continue;
      for (let x = 0; x < tied.length; x++) {
        for (let y = x + 1; y < tied.length; y++) {
          const a = tied[x];
          const b = tied[y];
          for (const [ga, gb] of h2hGames(a.p, b.p) ?? []) {
            a.h2h += ga > gb ? pts[0] : ga === gb ? pts[1] : pts[2];
            b.h2h += gb > ga ? pts[0] : ga === gb ? pts[1] : pts[2];
          }
        }
      }
    }
  }
  table.sort((u, v) => {
    for (const t of chain) {
      let c = 0;
      if (t === "points") c = v.points - u.points;
      else if (t === "h2h") c = v.h2h - u.h2h;
      else if (t === "gd") c = v.gf - v.ga - (u.gf - u.ga);
      else if (t === "gf") c = v.gf - u.gf;
      else if (t === "wins") c = v.wins - u.wins;
      else if (t === "seed") c = u.seed - v.seed;
      else c = u.lot - v.lot;
      if (c !== 0) return c;
    }
    return u.seed - v.seed;
  });
}

/** Spread seeded entrants over G groups (snake: 1..G, G..1, ...; block: consecutive). */
export function assignGroups<T>(order: T[], G: number, seeding: "snake" | "block" = "snake"): T[][] {
  const groups: T[][] = Array.from({ length: G }, () => []);
  const per = Math.ceil(order.length / G);
  order.forEach((p, i) => {
    let g: number;
    if (seeding === "block") g = Math.floor(i / per);
    else {
      const round = Math.floor(i / G);
      const pos = i % G;
      g = round % 2 === 0 ? pos : G - 1 - pos;
    }
    groups[g].push(p);
  });
  return groups;
}

export function nextPow2(m: number): number {
  let s = 1;
  while (s < m) s *= 2;
  return s;
}

/** Standard bracket order: seeds per slot so that 1 and 2 can only meet in the final. */
export function bracketOrder(size: number): number[] {
  let order = [1];
  while (order.length < size) {
    const n = order.length * 2;
    order = order.flatMap((s) => [s, n + 1 - s]);
  }
  return order;
}

export function roundWins(winsNeeded: number[], rounds: number): number[] {
  const len = winsNeeded.length;
  return Array.from({ length: rounds }, (_, r) => {
    const k = len - rounds + r;
    return winsNeeded[Math.max(0, k)];
  });
}

export function roundNames(rounds: number): string[] {
  const names = ["Winner", "Final", "Semifinal", "Quarterfinal"];
  return Array.from({ length: rounds + 1 }, (_, r) => {
    const fromEnd = rounds - r;
    return names[fromEnd] ?? `Round of ${2 ** fromEnd}`;
  });
}

function pairKey(stage: string, idA: number, idB: number): string {
  return idA < idB ? `${stage}|${idA}|${idB}` : `${stage}|${idB}|${idA}`;
}

function logKey(stage: string, a: number, b: number): string {
  return a < b ? `${stage}|${a}|${b}` : `${stage}|${b}|${a}`;
}

function initStats(s: StageConfig, n: number): StageStats {
  const st: StageStats = { id: s.id, type: s.type, entered: new Array(n).fill(0) };
  if (s.type === "knockout") {
    // Sized for the largest possible bracket; trimmed by the caller if needed.
    const R = Math.ceil(Math.log2(Math.max(2, n)));
    st.reached = Array.from({ length: n }, () => new Array(R + 1).fill(0));
    st.roundNames = roundNames(R);
  } else {
    st.pointsSum = new Array(n).fill(0);
    st.gamesSum = new Array(n).fill(0);
    if (s.type === "groups") st.position = Array.from({ length: n }, () => new Array(n).fill(0));
  }
  return st;
}

function validate(stages: StageConfig[]) {
  const seen = new Set(["entry"]);
  for (const s of stages) {
    for (const src of s.source) if (!seen.has(src.from)) throw new Error(`Stage ${s.id}: unknown source ${src.from}`);
    if (seen.has(s.id)) throw new Error(`Duplicate stage id ${s.id}`);
    seen.add(s.id);
  }
}

/** Convenience wrapper. */
export function simulateTournament(players: Player[], config: { stages: StageConfig[]; played?: PlayedGame[] }, model: GameModelJson, opts: SimOptions): SimResult {
  return new Tournament(players, config.stages, model, config.played).simulate(opts);
}
