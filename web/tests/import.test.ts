import { describe, expect, it } from "vitest";
import type { RankedPlayer } from "../src/data";
import { Tournament } from "../src/engine/simulate";
import { convertTournament, norm, tournamentIdFromLink, type EuStage, type EuTournament } from "../src/import/tableHockeyEu";
import { toyModel } from "./fixtures";

// Shaped like table-hockey.eu tournament 33: a 10-player league, A playoff (1-4) with a
// bronze band, B playoff (5-8), and a for-fun stage that is ignored.
const order = [29, 72, 333, 224, 81, 1532, 1589, 4501, 7000, 6351];
const names: Record<number, string> = {
  29: "Magnus Klippen", 72: "Andreas Fjermestad", 333: "Sinus Moberg", 224: "Benjamin Nygard", 81: "Amund Risa Fylling",
  1532: "Ruben Nesse Haland", 1589: "Sille Moberg", 4501: "Alex Lauritsen", 7000: "Olivia Vestre", 6351: "Tollak Nesse Haland",
};
const tournament: EuTournament = {
  id: 33, name: "Jærliga 2026/27 #12", level: "Level5", status: "InProgress",
  players: order.map((id, i) => ({ id, name: names[id], ithfId: id === 29 ? 660648 : null, seedPoints: 1000 - i })),
  stages: [{ id: 190, name: "Munnspill" }, { id: 191, name: "A" }, { id: 192, name: "B" }, { id: 193, name: "For fun", isForFun: true }],
};
const league: EuStage = {
  id: 190, name: "Munnspill", type: "RoundRobin", groupCount: 1, cycles: 1, scoringSystem: "Standard",
  field: { source: "Roster", rule: [] },
  groups: [{
    label: "A",
    players: order.map((playerId, i) => ({ playerId, seedPosition: i + 1 })),
    // Higher in the order always wins 3-1.
    matches: order.flatMap((a, i) => order.slice(i + 1).map((b) => ({ homePlayerId: a, awayPlayerId: b, homeScore: 3, awayScore: 1, walkover: "None" }))),
  }],
};
const series = (home: number, away: number, games: [number, number][]) => ({
  homePlayerId: home, awayPlayerId: away, walkover: "None",
  games: games.map(([h, a], k) => ({ gameNumber: k + 1, homeScore: h, awayScore: a, wentToOvertime: false, isBeyondClinch: false })),
});
const aPlayoff: EuStage = {
  id: 191, name: "A", type: "Playoff", divisionRank: 1,
  field: { source: "Stage", rule: [{ stageId: 190, groups: [], placeFrom: 1, placeTo: 4 }] },
  placementBands: [{ startPlace: 3, size: 2, sourceBandStartPlace: 1 }],
  playoffRounds: [
    { roundNumber: 1, bandStartPlace: 1, bandSize: 4, gamesToWin: 2, matches: [series(29, 224, [[5, 2], [4, 2]]), series(72, 333, [[1, 2], [2, 3]])] },
    { roundNumber: 2, bandStartPlace: 1, bandSize: 4, gamesToWin: 2, matches: [series(29, 333, [[7, 1], [5, 1]])] },
  ],
};
const bPlayoff: EuStage = {
  id: 192, name: "B", type: "Playoff", divisionRank: 2,
  field: { source: "Stage", rule: [{ stageId: 190, groups: [], placeFrom: 5, placeTo: 8 }] },
  playoffRounds: [],
};
const rated: RankedPlayer[] = [
  { id: 1647, name: "Magnus Klippen", country: "Norway", ithfId: 660648, worldRank: 30, rankingPoints: 3859, mu: 3, sd: 0.2, games: 4000, last: "2026-10-28" },
  { id: 9001, name: "Amund Risa Fylling", country: "Norway", ithfId: null, worldRank: null, rankingPoints: null, mu: 1, sd: 0.3, games: 900, last: "2026-10-28" },
];
const defaults = { prior_mean: -0.8, prior_sd: 0.9, level_prior: 0.5 };

describe("table-hockey.eu import", () => {
  it("matches names across accents and ASCII spellings", () => {
    expect(norm("Olivia Sørstrømmen Vestre")).toBe(norm("Olivia Sorstrommen Vestre"));
    expect(norm("Benjamin Nygård")).toBe(norm("Benjamin Nygard"));
  });

  it("reads tournament links", () => {
    expect(tournamentIdFromLink("https://table-hockey.eu/tournaments/33")).toBe(33);
    expect(tournamentIdFromLink("table-hockey.eu/tournaments/33?tab=players")).toBe(33);
    expect(tournamentIdFromLink("33")).toBe(33);
    expect(tournamentIdFromLink("https://example.com/33x")).toBeNull();
  });

  const res = convertTournament(tournament, [league, aPlayoff, bPlayoff], rated, defaults);

  it("maps players by ITHF id, then by name, and rates the rest as newcomers", () => {
    expect(res.entries.map((p) => p.id).slice(0, 5)).toEqual([1647, -72, -333, -224, 9001]);
    expect(res.newcomers).toHaveLength(8);
    // level 5: two steps below level 3
    expect(res.entries[1].mu).toBeCloseTo(-1.8);
  });

  it("rebuilds the real format and the games already played", () => {
    expect(res.stages.map((s) => [s.id, s.type, s.source, s.places])).toEqual([
      ["190", "groups", [{ from: "entry" }], undefined],
      ["191", "knockout", [{ from: "190", ranks: [1, 4] }], 1],
      ["192", "knockout", [{ from: "190", ranks: [5, 8] }], 5],
    ]);
    const a = res.stages[1];
    expect(a.type === "knockout" && a.winsNeeded).toEqual([2, 2]);
    expect(a.type === "knockout" && a.bronze).toBe(true);
    expect(res.played.filter((g) => g.stage === "190")).toHaveLength(45);
    expect(res.played.filter((g) => g.stage === "191")).toHaveLength(6);
    expect(res.notes.some((n) => n.includes("not drawn yet"))).toBe(true);
  });

  it("simulates only what is still open", () => {
    const players = res.entries.map((p) => ({ id: p.id, name: p.name, mu: p.mu, sd: p.sd }));
    const sim = new Tournament(players, res.stages, toyModel(), res.played).simulate({ runs: 300, seed: 1 });
    const idx = (id: number) => players.findIndex((p) => p.id === id);
    expect(sim.placeCounts[idx(1647)][0]).toBe(300); // won the final
    expect(sim.placeCounts[idx(-333)][1]).toBe(300); // lost the final
    // bronze between the semifinal losers is still open
    expect(sim.placeCounts[idx(-72)][2] + sim.placeCounts[idx(-224)][2]).toBe(300);
    // places 9-10 come from the finished league
    expect(sim.placeCounts[idx(-7000)][8]).toBe(300);
    expect(sim.placeCounts[idx(-6351)][9]).toBe(300);
  });
});
