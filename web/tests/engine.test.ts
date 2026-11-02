import { describe, expect, it } from "vitest";
import { Rng } from "../src/engine/rng";
import { assignGroups, bracketOrder, rankTable, roundWins, simulateTournament, Tournament } from "../src/engine/simulate";
import type { StageConfig } from "../src/engine/types";
import { players, toyModel } from "./fixtures";

const model = toyModel();

describe("helpers", () => {
  it("builds a standard bracket", () => {
    expect(bracketOrder(8)).toEqual([1, 8, 4, 5, 2, 7, 3, 6]);
  });
  it("fills every group with block seeding", () => {
    const sizes = assignGroups([...Array(10).keys()], 6, "block").map((g) => g.length);
    expect(sizes).toEqual([2, 2, 1, 2, 2, 1]);
  });
  it("aligns wins per round to the final", () => {
    expect(roundWins([3, 4, 4], 4)).toEqual([3, 3, 4, 4]);
    expect(roundWins([2, 3, 3, 4], 2)).toEqual([3, 4]);
  });
});

describe("group table tiebreaks", () => {
  // Real-style 4-player table: A and B tie on points, B beat A head to head.
  const games = new Map<string, [number, number][]>([
    ["0|1", [[1, 2]]], // A-B 1-2
    ["0|2", [[5, 0]]], // A-C 5-0
    ["0|3", [[3, 0]]], // A-D 3-0
    ["1|2", [[1, 1]]], // B-C 1-1
    ["1|3", [[2, 1]]], // B-D 2-1
    ["2|3", [[2, 2]]], // C-D 2-2
  ]);
  const results = (a: number, b: number): [number, number][] | undefined =>
    a < b ? games.get(`${a}|${b}`) : games.get(`${b}|${a}`)?.map(([u, v]) => [v, u] as [number, number]);
  const mk = (p: number, points: number, gf: number, ga: number) => ({ p, seed: p, points, gf, ga, wins: 0, games: 3, h2h: 0, lot: 0 });
  const table = () => [mk(0, 4, 9, 2), mk(1, 5, 5, 4), mk(2, 2, 3, 8), mk(3, 1, 3, 7)];

  it("ranks by points first", () => {
    const t = table();
    rankTable(t, ["points", "h2h", "gd", "gf", "lot"], results, [2, 1, 0]);
    expect(t.map((s) => s.p)).toEqual([1, 0, 2, 3]);
  });
  it("uses head-to-head among players tied on points before goal difference", () => {
    const t = table();
    t[1].points = 4; // A and B now level on 4
    rankTable(t, ["points", "h2h", "gd", "gf", "lot"], results, [2, 1, 0]);
    expect(t.map((s) => s.p)).toEqual([1, 0, 2, 3]);
    rankTable(t, ["points", "gd", "gf", "lot"], results, [2, 1, 0]);
    expect(t.map((s) => s.p)).toEqual([0, 1, 2, 3]);
  });
});

describe("tournament engine", () => {
  const groupsThenKo: StageConfig[] = [
    { id: "groups", type: "groups", groups: 2, source: [{ from: "entry" }], places: 1 },
    { id: "ko", type: "knockout", winsNeeded: [2, 3, 4], bronze: true, source: [{ from: "groups", ranks: [1, 8] }], places: 1 },
  ];

  it("gives every player exactly one place per run", () => {
    const ps = players(12);
    const res = simulateTournament(ps, { stages: groupsThenKo }, model, { runs: 2000, seed: 7 });
    for (let k = 0; k < ps.length; k++) {
      expect(res.placeCounts.reduce((s, row) => s + row[k], 0)).toBe(2000);
    }
    for (const row of res.placeCounts) expect(row.reduce((a, b) => a + b, 0)).toBe(2000);
    // only the 8 knockout players can finish in the top 8
    const ko = res.stages[1];
    for (let i = 0; i < ps.length; i++) {
      const top8 = res.placeCounts[i].slice(0, 8).reduce((a, b) => a + b, 0);
      expect(top8).toBe(ko.entered[i]);
    }
  });

  it("favours stronger players and stays reproducible with a seed", () => {
    const ps = players(16, 0.6);
    const cfg: StageConfig[] = [{ id: "ko", type: "knockout", winsNeeded: [3], source: [{ from: "entry" }], places: 1 }];
    const a = simulateTournament(ps, { stages: cfg }, model, { runs: 3000, seed: 1 });
    const b = simulateTournament(ps, { stages: cfg }, model, { runs: 3000, seed: 1 });
    expect(a.placeCounts).toEqual(b.placeCounts);
    expect(a.placeCounts[0][0]).toBeGreaterThan(a.placeCounts[15][0] * 5);
    // seeds 1 and 2 meet only in the final
    expect(a.finals["100-101"]).toBeGreaterThan(0);
  });

  it("gives top seeds byes when the bracket is not full", () => {
    const ps = players(6);
    const cfg: StageConfig[] = [{ id: "ko", type: "knockout", winsNeeded: [1], source: [{ from: "entry" }], places: 1 }];
    const res = simulateTournament(ps, { stages: cfg }, model, { runs: 500, seed: 3 });
    const reached = res.stages[0].reached!;
    const names = res.stages[0].roundNames!;
    // 6 players -> 8-bracket: QF, SF, F, winner occupy the last 4 slots
    const sf = names.indexOf("Semifinal");
    expect(reached[0][sf - 1]).toBe(500); // everyone with a slot "reaches" the quarterfinal
    // seeds 1 and 2 skip the first round (byes) so always reach the semifinal
    expect(reached[0][sf]).toBe(500);
    expect(reached[1][sf]).toBe(500);
    expect(reached[2][sf]).toBeLessThan(500);
    const won = reached.reduce((t, row) => t + row[row.length - 1], 0);
    expect(won).toBe(500);
  });

  it("fixes games already played (live mode)", () => {
    const ps = players(2, 0.01);
    const cfg: StageConfig[] = [{ id: "final", type: "knockout", winsNeeded: [2], source: [{ from: "entry" }], places: 1 }];
    const played = [
      { stage: "final", p1: 101, p2: 100, g1: 3, g2: 1 },
      { stage: "final", p1: 101, p2: 100, g1: 2, g2: 0 },
    ];
    const res = new Tournament(ps, cfg, model, played).simulate({ runs: 200, seed: 9 });
    expect(res.placeCounts[1][0]).toBe(200);
  });

  it("applies head-to-head correctly whatever the group order (live group table)", () => {
    const ps = players(4); // ids 100 A, 101 B, 102 C, 103 D
    const cfg: StageConfig[] = [{ id: "g", type: "groups", groups: 1, groupsExplicit: [[103, 102, 101, 100]], source: [{ from: "entry" }], places: 1 }];
    const played = [
      { stage: "g", p1: 100, p2: 101, g1: 0, g2: 1 }, // B beats A
      { stage: "g", p1: 100, p2: 102, g1: 5, g2: 0 },
      { stage: "g", p1: 101, p2: 102, g1: 0, g2: 1 },
      { stage: "g", p1: 100, p2: 103, g1: 0, g2: 0 },
      { stage: "g", p1: 101, p2: 103, g1: 0, g2: 0 },
      { stage: "g", p1: 102, p2: 103, g1: 0, g2: 1 },
    ];
    const res = new Tournament(ps, cfg, model, played).simulate({ runs: 20, seed: 1 });
    // D 4 pts, then A and B on 3: B won the meeting, so B is 2nd despite A's better goal difference
    expect(res.placeCounts[3][0]).toBe(20);
    expect(res.placeCounts[1][1]).toBe(20);
    expect(res.placeCounts[0][2]).toBe(20);
    expect(res.placeCounts[2][3]).toBe(20);
  });

  it("carries results into a later group whatever the pair order", () => {
    const ps = players(2, 0.01);
    const cfg: StageConfig[] = [
      { id: "g1", type: "groups", groups: 1, groupsExplicit: [[100, 101]], source: [{ from: "entry" }] },
      { id: "g2", type: "groups", groups: 1, groupsExplicit: [[101, 100]], carryFrom: "g1", source: [{ from: "g1", ranks: [1, 2] }], places: 1 },
    ];
    const played = [{ stage: "g1", p1: 100, p2: 101, g1: 3, g2: 0 }];
    const res = new Tournament(ps, cfg, model, played).simulate({ runs: 50, seed: 2 });
    expect(res.placeCounts[0][0]).toBe(50);
  });

  it("keeps played Swiss pairings (live mode)", () => {
    const ps = players(4, 0.01);
    const cfg: StageConfig[] = [{ id: "s", type: "swiss", rounds: 1, source: [{ from: "entry" }], places: 1 }];
    const played = [
      { stage: "s", p1: 100, p2: 102, g1: 0, g2: 2 },
      { stage: "s", p1: 101, p2: 103, g1: 0, g2: 2 },
    ];
    const res = new Tournament(ps, cfg, model, played).simulate({ runs: 50, seed: 4 });
    // 102 and 103 won their games: they take the top two places every time
    expect(res.placeCounts[2][0] + res.placeCounts[2][1]).toBe(50);
    expect(res.placeCounts[3][0] + res.placeCounts[3][1]).toBe(50);
  });

  it("runs a Swiss stage and placement brackets", () => {
    const ps = players(20);
    const cfg: StageConfig[] = [
      { id: "swiss", type: "swiss", rounds: 5, source: [{ from: "entry" }], places: 1 },
      { id: "a", type: "knockout", winsNeeded: [2], source: [{ from: "swiss", ranks: [1, 8] }], places: 1 },
      { id: "b", type: "knockout", winsNeeded: [1], source: [{ from: "swiss", ranks: [9, 16] }], places: 9 },
    ];
    const res = simulateTournament(ps, { stages: cfg }, model, { runs: 1000, seed: 5 });
    // only the title bracket's finals are reported, not the B playoff's
    expect(Object.values(res.finals).reduce((a, b) => a + b, 0)).toBe(1000);
    for (let i = 0; i < ps.length; i++) {
      const inB = res.stages[2].entered[i];
      const placesB = res.placeCounts[i].slice(8, 16).reduce((a, b) => a + b, 0);
      expect(placesB).toBe(inB);
    }
  });

  it("simulates 10,000 runs of a 100-player event in a few seconds", () => {
    const ps = players(100, 0.05);
    const cfg: StageConfig[] = [
      { id: "groups", type: "groups", groups: 10, source: [{ from: "entry" }], places: 1 },
      { id: "ko", type: "knockout", winsNeeded: [3, 3, 4, 4, 4], bronze: true, source: [{ from: "groups", ranks: [1, 32] }], places: 1 },
    ];
    const t = Date.now();
    simulateTournament(ps, { stages: cfg }, model, { runs: 10000, seed: 2 });
    expect(Date.now() - t).toBeLessThan(15000);
  }, 30000);

  it("draws a normal sample with unit variance", () => {
    const r = new Rng(4);
    let s = 0;
    let s2 = 0;
    for (let i = 0; i < 50000; i++) {
      const z = r.normal();
      s += z;
      s2 += z * z;
    }
    expect(Math.abs(s / 50000)).toBeLessThan(0.02);
    expect(Math.abs(s2 / 50000 - 1)).toBeLessThan(0.03);
  });
});
