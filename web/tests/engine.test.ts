import { describe, expect, it } from "vitest";
import { Gbm, pairShifts } from "../src/engine/gbm";
import { dayFormPosterior, drawPosterior } from "../src/engine/dayForm";
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

  it("applies head-to-head edges to the named pair only", () => {
    const ps = players(2, 0);
    const cfg: StageConfig[] = [{ id: "ko", type: "knockout", winsNeeded: [1], source: [{ from: "entry" }], places: 1 }];
    const even = simulateTournament(ps, { stages: cfg }, model, { runs: 4000, seed: 5 });
    const edged = simulateTournament(ps, { stages: cfg, h2h: { "100-101": 1.0 } }, model, { runs: 4000, seed: 5 });
    const flipped = simulateTournament(ps, { stages: cfg, h2h: { "100-101": -1.0 } }, model, { runs: 4000, seed: 5 });
    expect(Math.abs(even.placeCounts[0][0] - 2000)).toBeLessThan(200);
    expect(edged.placeCounts[0][0]).toBeGreaterThan(2600);
    expect(flipped.placeCounts[1][0]).toBeGreaterThan(2600);
    const other = simulateTournament(ps, { stages: cfg, h2h: { "100-999": 1.0 } }, model, { runs: 4000, seed: 5 });
    expect(other.placeCounts).toEqual(even.placeCounts);
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

describe("LightGBM layer", () => {
  // Two-feature toy model: class 0 (A wins) grows with d, class 2 (B wins) with -d, draws flat.
  const tree = (sign: number) => ({ f: [0], t: [0], l: [~0], r: [~1], m: [0], dl: [1], v: [-sign, sign], root: 0 });
  const json = {
    features: ["d", "x_1", "x_2"],
    side: ["x"],
    num_class: 3,
    trees: [tree(1), { f: [], t: [], l: [], r: [], m: [], dl: [], v: [0], root: ~0 }, tree(-1)],
  };

  it("evaluates trees and handles missing values", () => {
    const g = new Gbm(json);
    const up = g.predictRow(Float64Array.from([1, 0, 0]));
    const down = g.predictRow(Float64Array.from([-1, 0, 0]));
    expect(up[0]).toBeGreaterThan(up[2]);
    expect(down[2]).toBeGreaterThan(down[0]);
    expect(up[0] + up[1] + up[2]).toBeCloseTo(1, 12);
    // NaN with missing type "none" is treated as 0, which goes left (<= 0).
    expect(g.predictRow(Float64Array.from([NaN, 0, 0]))).toEqual(down);
  });

  it("gives order-free odds and antisymmetric pair shifts", () => {
    const g = new Gbm(json);
    const a = { id: 1, mu: 1, sd: 0.2, feat: [0] };
    const b = { id: 2, mu: 0, sd: 0.2, feat: [null] };
    const ab = g.pairOdds(a, b, 0, 0, { level: -1, bordshockey: 0 });
    const ba = g.pairOdds(b, a, 0, 0, { level: -1, bordshockey: 0 });
    expect(ab[0]).toBeCloseTo(ba[2], 12);
    expect(ab[1]).toBeCloseTo(ba[1], 12);
    const s = pairShifts(g, [b, a], {}, 0.8, { level: -1, bordshockey: 0 });
    expect(Object.keys(s)).toEqual(["1-2"]);
    expect(Number.isFinite(s["1-2"])).toBe(true);
  });
});

describe("same-day form (live mode)", () => {
  it("matches the closed form for one game between two players", () => {
    // prior N(0, 1) each, one game a beats b 3-0 with noise sd 2: shift of a = v * gd / (2v + s^2)
    const post = dayFormPosterior([0, 0], [1, 1], [{ a: 0, b: 1, gd: 3, shift: 0 }], 2, 6);
    expect(post.mean[0]).toBeCloseTo(3 / 6, 10);
    expect(post.mean[1]).toBeCloseTo(-3 / 6, 10);
    // draws have the posterior covariance: var = 1 - 1/6, cov = 1/6
    const rng = new Rng(3);
    const z = new Float64Array(2);
    const x = new Float64Array(2);
    let s00 = 0, s01 = 0;
    const N = 200000;
    for (let k = 0; k < N; k++) {
      z[0] = rng.normal(); z[1] = rng.normal();
      drawPosterior(post, z, x);
      s00 += (x[0] - 0.5) ** 2;
      s01 += (x[0] - 0.5) * (x[1] + 0.5);
    }
    expect(s00 / N).toBeCloseTo(5 / 6, 2);
    expect(s01 / N).toBeCloseTo(1 / 6, 2);
  });

  it("subtracts the pair shift and clips the goal difference", () => {
    const a = dayFormPosterior([0, 0], [1, 1], [{ a: 0, b: 1, gd: 20, shift: 1 }], 2, 6);
    expect(a.mean[0]).toBeCloseTo(5 / 6, 10);
  });

  it("lifts a player who won the day in the playoff odds, and only with played games", () => {
    const ps = players(4, 0).map((p) => ({ ...p, sd: 0.4 }));
    const cfg: StageConfig[] = [
      { id: "groups", type: "groups", groups: 1, source: [{ from: "entry" }], places: 1 },
      { id: "ko", type: "knockout", winsNeeded: [1, 1], source: [{ from: "groups", ranks: [1, 4] }], places: 1 },
    ];
    // P4 crushes everyone in the group; P1 loses every game narrowly
    const played = [
      { stage: "groups", p1: 103, p2: 100, g1: 6, g2: 0 },
      { stage: "groups", p1: 103, p2: 101, g1: 5, g2: 0 },
      { stage: "groups", p1: 103, p2: 102, g1: 6, g2: 1 },
      { stage: "groups", p1: 101, p2: 100, g1: 2, g2: 1 },
      { stage: "groups", p1: 102, p2: 100, g1: 2, g2: 1 },
      { stage: "groups", p1: 101, p2: 102, g1: 1, g2: 1 },
    ];
    const t = new Tournament(ps, cfg, model, played);
    const on = t.simulate({ runs: 20000, seed: 5 });
    const off = t.simulate({ runs: 20000, seed: 5, dayForm: false });
    expect(on.placeCounts[3][0]).toBeGreaterThan(off.placeCounts[3][0] + 600); // about +1,300 expected (sd ~100)
    // walkovers count for the table but not for form
    const wo = played.map((g) => ({ ...g, walkover: true }));
    const tw = new Tournament(ps, cfg, model, wo);
    expect(tw.simulate({ runs: 500, seed: 2 }).placeCounts).toEqual(tw.simulate({ runs: 500, seed: 2, dayForm: false }).placeCounts);
    // no played games: identical to the old draw
    const t0 = new Tournament(ps, cfg, model);
    expect(t0.simulate({ runs: 500, seed: 2 }).placeCounts).toEqual(t0.simulate({ runs: 500, seed: 2, dayForm: false }).placeCounts);
  });
});
