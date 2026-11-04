/**
 * Import a tournament from table-hockey.eu: entry list, real stage format and games
 * already played (live mode), from its public JSON API.
 */
import type { RankedPlayer } from "../data";
import type { EntrySource, PlayedGame, StageConfig } from "../engine/types";

export const API = "https://table-hockey.eu/api";

// --- API shapes (only the fields used here) -------------------------------------------

export interface EuPlayer {
  id: number;
  ithfId?: number | null;
  name: string;
  nation?: string;
  seedPoints?: number | null;
  isWithdrawn?: boolean;
}

export interface EuStageRef {
  id: number;
  name: string;
  isForFun?: boolean;
}

export interface EuTournament {
  id: number;
  name: string;
  level?: string | null;
  status?: string;
  startDate?: string;
  players: EuPlayer[];
  stages: EuStageRef[];
}

interface EuMatch {
  homePlayerId?: number | null;
  awayPlayerId?: number | null;
  homeScore?: number | null;
  awayScore?: number | null;
  isBye?: boolean;
  walkover?: string;
  wentToOvertime?: boolean;
}

interface EuSeries extends EuMatch {
  position?: number;
  games?: { gameNumber: number; homeScore?: number | null; awayScore?: number | null; wentToOvertime?: boolean; isBeyondClinch?: boolean }[];
}

export interface EuStage {
  id: number;
  name: string;
  type: string;
  isForFun?: boolean;
  groupCount?: number;
  cycles?: number;
  scoringSystem?: string;
  divisionRank?: number;
  groups?: { label: string; players: { playerId: number; seedPosition?: number }[]; matches: EuMatch[] }[];
  playoffRounds?: { roundNumber: number; bandStartPlace: number; bandSize: number; gamesToWin: number; matches?: EuSeries[] }[];
  placementBands?: { startPlace: number; size: number; sourceBandStartPlace: number }[];
  field?: { source?: string; rule?: { stageId: number; groups?: unknown[]; placeFrom: number; placeTo: number; isRoster?: boolean }[] };
}

export function tournamentIdFromLink(link: string): number | null {
  const m = link.trim().match(/table-hockey\.eu\/(?:api\/)?tournaments\/(\d+)/i) ?? link.trim().match(/^(\d+)$/);
  return m ? Number(m[1]) : null;
}

export async function fetchTournament(id: number, fetcher: typeof fetch = fetch): Promise<{ tournament: EuTournament; stages: EuStage[] }> {
  const get = async <T,>(path: string): Promise<T> => {
    const r = await fetcher(`${API}/${path}`, { headers: { Accept: "application/json" } });
    if (!r.ok) throw new Error(`table-hockey.eu answered ${r.status} for ${path}`);
    return r.json() as Promise<T>;
  };
  const tournament = await get<EuTournament>(`tournaments/${id}`);
  const stages = await Promise.all(tournament.stages.filter((s) => !s.isForFun).map((s) => get<EuStage>(`stages/${s.id}`)));
  return { tournament, stages };
}

// --- conversion ----------------------------------------------------------------------

export interface RatingDefaults {
  prior_mean: number;
  prior_sd: number;
  level_prior: number;
}

export interface ImportResult {
  name: string;
  status: string;
  /** Entry list in seed order, as rated players (unknown players get a newcomer rating). */
  entries: RankedPlayer[];
  /** Players with no rating history: not in the database. */
  newcomers: string[];
  stages: StageConfig[];
  played: PlayedGame[];
  notes: string[];
  /** Event level from table-hockey.eu (1 = majors .. 6), 0 when unknown. */
  level: number;
}

/** Name key that survives ASCII spellings: "Sørstrømmen" and "Sorstrommen" match. */
const TRANSLIT: Record<string, string> = { ø: "o", æ: "ae", œ: "oe", ß: "ss", ł: "l", đ: "d", þ: "th", ı: "i" };
export const norm = (s: string) =>
  s
    .toLowerCase()
    .replace(/[øæœßłđþı]/g, (c) => TRANSLIT[c])
    .normalize("NFD")
    .replace(/[̀-ͯ]/g, "")
    .toLowerCase()
    .replace(/[^a-z]+/g, " ")
    .trim();

/** Walkovers count as 5-0, the usual table hockey walkover score. */
function walkoverScore(w: string | undefined): [number, number] | null {
  if (!w || w === "None") return null;
  if (/home/i.test(w)) return [5, 0];
  if (/away/i.test(w)) return [0, 5];
  return null;
}

export function convertTournament(
  t: EuTournament,
  stages: EuStage[],
  rated: RankedPlayer[],
  defaults: RatingDefaults,
): ImportResult {
  const notes: string[] = [];
  const byIthf = new Map(rated.filter((p) => p.ithfId != null).map((p) => [p.ithfId!, p]));
  const byName = new Map<string, RankedPlayer | null>();
  for (const p of rated) {
    const k = norm(p.name);
    byName.set(k, byName.has(k) ? null : p); // ambiguous names never match
  }
  const level = Number(String(t.level ?? "").replace(/\D+/g, "")) || 0;
  const newcomerMu = defaults.prior_mean + (level > 0 ? -defaults.level_prior * (level - 3) : 0);

  // Entry list: active players by seed points (the site's own seeding), unknown last.
  const roster = t.players
    .filter((p) => !p.isWithdrawn)
    .slice()
    .sort((a, b) => (b.seedPoints ?? -1) - (a.seedPoints ?? -1));
  const idOf = new Map<number, number>(); // table-hockey.eu player id -> simulator id
  const newcomers: string[] = [];
  const entries: RankedPlayer[] = roster.map((p) => {
    const hit = (p.ithfId != null ? byIthf.get(p.ithfId) : undefined) ?? byName.get(norm(p.name)) ?? undefined;
    if (hit) {
      idOf.set(p.id, hit.id);
      return hit;
    }
    newcomers.push(p.name);
    const id = -p.id; // never clashes with database ids
    idOf.set(p.id, id);
    return {
      id, name: p.name, country: p.nation ?? "", ithfId: p.ithfId ?? null, worldRank: null, rankingPoints: null,
      mu: newcomerMu, sd: defaults.prior_sd, games: 0, last: "",
    };
  });
  const sid = (id: number) => String(id);
  const known = new Set(stages.map((s) => s.id));
  const groupCount = new Map(stages.map((s) => [s.id, s.groups?.length || s.groupCount || 1]));
  const mapped = (euId: number | null | undefined) => (euId != null ? idOf.get(euId) : undefined);

  const out: StageConfig[] = [];
  const played: PlayedGame[] = [];
  for (const s of stages) {
    // Entrants: the roster, or slices of earlier stages' rankings. Places are per group on the
    // site; the engine ranks all group winners first, so per-group places p..q map to overall ranks.
    const rules = (s.field?.rule ?? []).filter((r) => !r.isRoster && known.has(r.stageId));
    const source: EntrySource[] = rules.length
      ? rules.map((r) => {
          const g = groupCount.get(r.stageId) ?? 1;
          if (r.groups && r.groups.length) notes.push(`${s.name}: entrants taken from specific groups are approximated by overall ranks.`);
          return { from: sid(r.stageId), ranks: [(r.placeFrom - 1) * g + 1, r.placeTo * g] as [number, number] };
        })
      : [{ from: "entry" }];
    const firstPlace = rules.length === 1 ? (rules[0].placeFrom - 1) * (groupCount.get(rules[0].stageId) ?? 1) + 1 : 1;
    if (s.scoringSystem && s.scoringSystem !== "Standard") notes.push(`${s.name}: scoring "${s.scoringSystem}" simulated as 2-1-0.`);

    if (s.type === "RoundRobin") {
      const groups = (s.groups ?? []).map((g) =>
        g.players
          .slice()
          .sort((a, b) => (a.seedPosition ?? 0) - (b.seedPosition ?? 0))
          .map((p) => mapped(p.playerId))
          .filter((x): x is number => x !== undefined),
      );
      out.push({
        id: sid(s.id), name: s.name, type: "groups", groups: Math.max(1, s.groupCount ?? groups.length ?? 1), legs: s.cycles ?? 1,
        source, ...(groups.length && groups.every((g) => g.length) ? { groupsExplicit: groups } : {}),
      });
      for (const g of s.groups ?? [])
        for (const m of g.matches ?? []) {
          const a = mapped(m.homePlayerId);
          const b = mapped(m.awayPlayerId);
          if (a === undefined || b === undefined || m.isBye) continue;
          const wo = walkoverScore(m.walkover);
          const score: [number, number] | null = wo ?? (m.homeScore != null && m.awayScore != null ? [m.homeScore, m.awayScore] : null);
          if (score) played.push({ stage: sid(s.id), p1: a, p2: b, g1: score[0], g2: score[1], ot: !!m.wentToOvertime, ...(wo ? { walkover: true } : {}) });
        }
    } else if (s.type === "Playoff") {
      // The championship band: rounds whose band starts at the stage's first place.
      const rounds = (s.playoffRounds ?? []).filter((r) => r.bandStartPlace === 1).sort((a, b) => a.roundNumber - b.roundNumber);
      let winsNeeded = rounds.map((r) => r.gamesToWin);
      if (!winsNeeded.length) {
        winsNeeded = [2];
        notes.push(`${s.name}: the bracket is not drawn yet, so series are simulated as first to 2.`);
      }
      const bronze = (s.placementBands ?? []).some((b) => b.startPlace === 3 && b.size === 2 && b.sourceBandStartPlace === 1);
      // The drawn bracket: first-round series in position order, home then away.
      const first = (rounds[0]?.matches ?? []).slice().sort((x, y) => (x.position ?? 0) - (y.position ?? 0));
      const draw = first.flatMap((m) => [m.homePlayerId, m.awayPlayerId].map((id) => mapped(id) ?? null));
      out.push({
        id: sid(s.id), name: s.name, type: "knockout", winsNeeded, source, places: firstPlace,
        ...(draw.some((x) => x !== null) ? { draw } : {}),
        ...(bronze ? { bronze: true, bronzeWins: winsNeeded[winsNeeded.length - 1] } : {}),
      });
      // Championship games go to the bracket, 3rd-4th games to the engine's bronze series; other bands are not simulated.
      for (const r of s.playoffRounds ?? []) {
        const stageKey = r.bandStartPlace === 1 ? sid(s.id) : bronze && r.bandStartPlace === 3 && r.bandSize === 2 ? `${s.id}:bronze` : null;
        if (!stageKey) continue;
        for (const m of r.matches ?? []) {
          const a = mapped(m.homePlayerId);
          const b = mapped(m.awayPlayerId);
          if (a === undefined || b === undefined || m.isBye) continue;
          const wo = walkoverScore(m.walkover);
          if (wo) {
            // A walked-over series: the winner takes the games it needed.
            const need = r.gamesToWin;
            for (let k = 0; k < need; k++)
              played.push({ stage: stageKey, p1: a, p2: b, g1: wo[0], g2: wo[1], walkover: true });
            continue;
          }
          for (const g of (m.games ?? []).slice().sort((x, y) => x.gameNumber - y.gameNumber)) {
            if (g.isBeyondClinch || g.homeScore == null || g.awayScore == null) continue;
            played.push({ stage: stageKey, p1: a, p2: b, g1: g.homeScore, g2: g.awayScore, ot: !!g.wentToOvertime });
          }
        }
      }
    } else {
      notes.push(`${s.name}: stage type "${s.type}" is not supported and was left out.`);
    }
  }
  // A league-only event: its table is the final ranking.
  if (out.length && !out.some((s) => s.places)) out[out.length - 1].places = 1;
  if (newcomers.length) {
    notes.push(
      `${newcomers.length} player${newcomers.length > 1 ? "s have" : " has"} no rating history and start${newcomers.length > 1 ? "" : "s"} as a newcomer for a level ${level || "?"} event.`,
    );
  }
  return { name: t.name, status: t.status ?? "", entries, newcomers, stages: out, played, notes, level };
}
