/**
 * Same-day form in live mode: condition every player's strength on the games already played.
 *
 * Before the event each strength is Normal(mu, sd^2 + formSd^2): the rating posterior plus the
 * per-event form shock. A played game says clip(g1 - g2) = s_a - s_b + shift_ab + noise, with
 * noise sd `obsSd` (the Kalman rating's per-game goal-difference noise). The posterior over all
 * strengths is Gaussian again; with no games it is exactly the prior the engine always used.
 *
 *   precision L = diag(1 / v) + sum_g (e_a - e_b)(e_a - e_b)' / obsSd^2
 *   L mean      = mu / v       + sum_g (gd_g - shift_g)(e_a - e_b) / obsSd^2
 *
 * A draw is mean + R^-T z, where L = R R' (Cholesky, R lower) and z is standard normal.
 */

export interface DayFormGame {
  a: number; // player index
  b: number;
  gd: number; // goals a - b, overtime goal included
  shift: number; // head-to-head / LightGBM shift of a over b, rating units
}

export interface DayFormPosterior {
  n: number;
  mean: Float64Array;
  /** Lower Cholesky factor of the precision matrix, row-major n x n. */
  chol: Float64Array;
}

export function dayFormPosterior(
  mu: ArrayLike<number>,
  variance: ArrayLike<number>,
  games: DayFormGame[],
  obsSd: number,
  gdCap: number,
): DayFormPosterior {
  const n = mu.length;
  const L = new Float64Array(n * n);
  const h = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const v = Math.max(variance[i], 1e-9);
    L[i * n + i] = 1 / v;
    h[i] = mu[i] / v;
  }
  const w = 1 / (obsSd * obsSd);
  for (const g of games) {
    if (g.a === g.b) continue;
    const y = (Math.max(-gdCap, Math.min(gdCap, g.gd)) - g.shift) * w;
    L[g.a * n + g.a] += w;
    L[g.b * n + g.b] += w;
    L[g.a * n + g.b] -= w;
    L[g.b * n + g.a] -= w;
    h[g.a] += y;
    h[g.b] -= y;
  }
  // Cholesky in place (lower triangle).
  for (let j = 0; j < n; j++) {
    let d = L[j * n + j];
    for (let k = 0; k < j; k++) d -= L[j * n + k] * L[j * n + k];
    const r = Math.sqrt(d);
    L[j * n + j] = r;
    for (let i = j + 1; i < n; i++) {
      let s = L[i * n + j];
      for (let k = 0; k < j; k++) s -= L[i * n + k] * L[j * n + k];
      L[i * n + j] = s / r;
    }
    for (let k = j + 1; k < n; k++) L[j * n + k] = 0;
  }
  // mean = L^-1 h: forward (R y = h), then back (R' mean = y).
  const mean = forward(L, n, h);
  backInPlace(L, n, mean);
  return { n, mean, chol: L };
}

/** One posterior draw into `out`, using `z` (standard normals, overwritten). */
export function drawPosterior(post: DayFormPosterior, z: Float64Array, out: Float64Array): void {
  backInPlace(post.chol, post.n, z);
  for (let i = 0; i < post.n; i++) out[i] = post.mean[i] + z[i];
}

function forward(R: Float64Array, n: number, b: ArrayLike<number>): Float64Array {
  const x = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = b[i];
    for (let k = 0; k < i; k++) s -= R[i * n + k] * x[k];
    x[i] = s / R[i * n + i];
  }
  return x;
}

/** Solve R' x = b in place (R lower, so R' is upper). */
function backInPlace(R: Float64Array, n: number, x: Float64Array): void {
  for (let i = n - 1; i >= 0; i--) {
    let s = x[i];
    for (let k = i + 1; k < n; k++) s -= R[k * n + i] * x[k];
    x[i] = s / R[i * n + i];
  }
}
