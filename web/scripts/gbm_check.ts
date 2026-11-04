/** Check the TypeScript tree evaluator against LightGBM: tsx scripts/gbm_check.ts gbm.json check.json */
import { readFileSync } from "node:fs";
import { Gbm } from "../src/engine/gbm";

const [modelPath, checkPath] = process.argv.slice(2);
const gbm = new Gbm(JSON.parse(readFileSync(modelPath, "utf8")));
const check = JSON.parse(readFileSync(checkPath, "utf8")) as { X: (number | null)[][]; p: number[][] };
let worst = 0;
check.X.forEach((row, i) => {
  const p = gbm.predictRow(Float64Array.from(row.map((v) => (v === null ? NaN : v))));
  p.forEach((v, k) => (worst = Math.max(worst, Math.abs(v - check.p[i][k]))));
});
console.log(`rows ${check.X.length}, max |diff| ${worst.toExponential(2)}`);
if (worst > 1e-5) process.exit(1);
