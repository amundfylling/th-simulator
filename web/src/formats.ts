/** Turn the format builder's simple choices into an engine config. */
import { nextPow2, roundNames } from "./engine/simulate";
import type { StageConfig } from "./engine/types";

export type FormatKind = "groups_ko" | "ko" | "league" | "swiss_ko";

export interface FormatState {
  kind: FormatKind;
  groups: number;
  legs: number;
  advance: number;
  /** Wins needed per knockout round, first round first. */
  koWins: number[];
  bronze: boolean;
  swissRounds: number;
  bPlayoff: boolean;
  points: [number, number, number];
  /** How World Ranking seeds are spread over groups. Default snake. */
  seeding?: "snake" | "abba";
}

export const defaultFormat = (n: number): FormatState => ({
  kind: n <= 8 ? "ko" : "groups_ko",
  groups: Math.max(1, Math.round(n / 9)),
  legs: 1,
  // Largest power of two up to half the field, at most 16.
  advance: Math.min(16, Math.max(2, nextPow2(Math.floor(n / 2) + 1) / 2)),
  koWins: [],
  bronze: true,
  swissRounds: Math.max(3, Math.ceil(Math.log2(Math.max(2, n))) + 1),
  bPlayoff: false,
  points: [2, 1, 0],
  seeding: "snake",
});

export function knockoutSize(f: FormatState, n: number): number {
  return f.kind === "ko" ? n : f.kind === "league" ? 0 : Math.min(f.advance, n);
}

export function roundsFor(size: number): string[] {
  if (size < 2) return [];
  const R = Math.log2(nextPow2(size));
  return roundNames(R).slice(0, R);
}

/** Wins per round, filling defaults: early rounds first to 3, semis and final first to 4. */
export function koWinsFor(f: FormatState, size: number): number[] {
  const R = roundsFor(size).length;
  return Array.from({ length: R }, (_, r) => f.koWins[r] ?? (R - r <= 2 ? 4 : 3));
}

export function buildStages(f: FormatState, n: number): StageConfig[] {
  const koSize = knockoutSize(f, n);
  const wins = koWinsFor(f, koSize);
  const pre = f.kind === "groups_ko" ? "groups" : f.kind === "swiss_ko" ? "swiss" : null;
  const stages: StageConfig[] = [];
  if (f.kind === "groups_ko" || f.kind === "league") {
    stages.push({
      id: "groups",
      name: f.kind === "league" ? "League" : "Group stage",
      type: "groups",
      groups: f.kind === "league" ? 1 : Math.max(1, Math.min(f.groups, Math.floor(n / 2))),
      legs: f.legs,
      points: f.points,
      seeding: f.seeding ?? "snake",
      source: [{ from: "entry" }],
      places: 1,
    });
  }
  if (f.kind === "swiss_ko") {
    stages.push({ id: "swiss", name: "Swiss", type: "swiss", rounds: f.swissRounds, points: f.points, source: [{ from: "entry" }], places: 1 });
  }
  if (f.kind === "ko") {
    stages.push({ id: "ko", name: "Playoff", type: "knockout", winsNeeded: wins, bronze: f.bronze, source: [{ from: "entry" }], places: 1 });
  }
  if (pre && koSize >= 2) {
    stages.push({ id: "ko", name: "Playoff", type: "knockout", winsNeeded: wins, bronze: f.bronze, source: [{ from: pre, ranks: [1, koSize] }], places: 1 });
    if (f.bPlayoff && n > koSize + 1) {
      const bSize = Math.min(koSize, n - koSize);
      stages.push({
        id: "ko_b",
        name: "B playoff",
        type: "knockout",
        winsNeeded: koWinsFor({ ...f, koWins: [] }, bSize).map((w) => Math.min(w, 3)),
        source: [{ from: pre, ranks: [koSize + 1, koSize + bSize] }],
        places: koSize + 1,
      });
    }
  }
  return stages;
}
