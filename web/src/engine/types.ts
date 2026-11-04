/** Configuration and data types shared by the engine, the website and the backtests. */

export interface Player {
  id: number;
  name: string;
  country?: string;
  /** Rating mean and posterior sd in model units. */
  mu: number;
  sd: number;
  /** World Ranking position; group stages are always seeded by it. Unranked players come last, in entry order. */
  worldRank?: number | null;
}

export interface GameModelJson {
  regulation: { rr: { beta: number; c: number }; po: { beta: number; c: number } };
  overtime: { beta: number };
  scoreline: {
    x_bin: number;
    n_bins: number;
    beta: number;
    /** tables[kind][outcome][bin] = [favGoals, dogGoals, prob][]; kind rr = league games, po = playoff. */
    tables: Record<"rr" | "po", Record<"fav" | "draw" | "dog", [number, number, number][][]>>;
  };
  /** Per-player per-event form shock sd, in rating units. */
  form_sd?: number;
}

/** Where a stage's entrants come from: the entry list or another stage's ranking. */
export interface EntrySource {
  /** "entry" or the id of an earlier stage. */
  from: string;
  /** 1-based inclusive slice of that ranking; all when omitted. */
  ranks?: [number, number];
}

export type Tiebreak = "points" | "h2h" | "gd" | "gf" | "wins" | "seed" | "lot";

interface BaseStage {
  id: string;
  name?: string;
  source: EntrySource[];
  /** When set, this stage's ranking fixes final places starting here (e.g. 1, or 17 for a B playoff). */
  places?: number;
}

export interface GroupStageConfig extends BaseStage {
  type: "groups";
  groups: number;
  /** 1 = single round-robin, 2 = double. */
  legs?: number;
  /** Points for win, draw, loss. Default 2-1-0. */
  points?: [number, number, number];
  tiebreak?: Tiebreak[];
  /** How seeds are spread over groups. Seeds follow the World Ranking (see `worldRankOrder`). Default snake. */
  seeding?: "snake" | "abba" | "block" | "random";
  /** Fixed group composition as player ids (e.g. a historical draw); overrides `groups`/`seeding`. */
  groupsExplicit?: number[][];
  /** Carry over games between players who met in this earlier group stage. */
  carryFrom?: string;
}

export interface KnockoutStageConfig extends BaseStage {
  type: "knockout";
  /** Wins needed per round, first round first. Shorter arrays repeat their first value for early rounds. */
  winsNeeded: number[];
  bronze?: boolean;
  bronzeWins?: number;
  /** A drawn bracket: first-round slots as player ids in bracket order (null = bye). Used when it holds exactly the stage's entrants. */
  draw?: (number | null)[];
  /** Drawn games stand (no golden goal) and do not count as wins. */
  drawsAllowed?: boolean;
}

export interface SwissStageConfig extends BaseStage {
  type: "swiss";
  rounds: number;
  points?: [number, number, number];
  tiebreak?: Tiebreak[];
}

export type StageConfig = GroupStageConfig | KnockoutStageConfig | SwissStageConfig;

/** A game already played. In a Swiss stage, games fill the rounds in the order listed. */
export interface PlayedGame {
  stage: string;
  p1: number;
  p2: number;
  g1: number;
  g2: number;
  ot?: boolean;
}

export interface TournamentConfig {
  name?: string;
  stages: StageConfig[];
  /** Games already played (live mode). Player ids, not indices. */
  played?: PlayedGame[];
  /** Head-to-head edges: "idA-idB" (idA < idB) -> rating shift in idA's favour against idB. */
  h2h?: Record<string, number>;
}

export interface SimOptions {
  runs: number;
  seed?: number;
  /** Draw each player's rating from the posterior once per run. Default true. */
  ratingUncertainty?: boolean;
  formSd?: number;
  onProgress?: (done: number) => void;
}

export interface StageStats {
  id: string;
  type: StageConfig["type"];
  /** entered[i] = runs in which player i entered the stage. */
  entered: number[];
  /** knockout: reached[i][r] = runs in which player i reached round r, aligned to the end (last = won, before it = final). */
  reached?: number[][];
  roundNames?: string[];
  /** groups: position[i][k] = runs finishing k+1 in own group. */
  position?: number[][];
  /** groups/swiss: summed points (for expected points). */
  pointsSum?: number[];
  gamesSum?: number[];
}

export interface SimResult {
  runs: number;
  players: Player[];
  /** placeCounts[i][k] = runs in which player i finished k+1. */
  placeCounts: number[][];
  stages: StageStats[];
  /** "idA-idB" (sorted) -> count of title finals between them (placement brackets excluded). */
  finals: Record<string, number>;
  elapsedMs: number;
}
