import type { GameModelJson, Player } from "../src/engine/types";

/** A small, hand-made game model close to the fitted one (gap in goal units). */
export function toyModel(): GameModelJson {
  const nBins = 24;
  const bins = (pairs: [number, number][]) =>
    Array.from({ length: nBins }, () => pairs.map(([a, b]) => [a, b, 1] as [number, number, number]));
  return {
    regulation: { rr: { beta: 0.8, c: 0.37 }, po: { beta: 0.8, c: 0.37 } },
    overtime: { beta: 0.5 },
    scoreline: {
      x_bin: 0.25,
      n_bins: nBins,
      beta: 0.8,
      tables: {
        rr: { fav: bins([[1, 0], [2, 1], [3, 1], [4, 2]]), draw: bins([[0, 0], [1, 1], [2, 2]]), dog: bins([[0, 1], [1, 2], [1, 3]]) },
        po: { fav: bins([[1, 0], [2, 1], [3, 2]]), draw: bins([[1, 1], [2, 2]]), dog: bins([[0, 1], [1, 2]]) },
      },
    },
  };
}

export function players(n: number, spread = 0.5): Player[] {
  return Array.from({ length: n }, (_, i) => ({ id: 100 + i, name: `P${i + 1}`, mu: (n - i) * spread, sd: 0 }));
}
