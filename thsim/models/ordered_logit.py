"""Three-way regulation outcome model on a rating gap d.

P(A wins) = sigma(beta*d - c), P(B wins) = sigma(-beta*d - c), draw = the rest.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

# Gauss-Hermite nodes for integrating over rating uncertainty (probabilists' version).
GH_X, GH_W = np.polynomial.hermite_e.hermegauss(7)
GH_W = GH_W / GH_W.sum()


def probs(d: np.ndarray, beta: float, c: float) -> np.ndarray:
    pa = expit(beta * d - c)
    pb = expit(-beta * d - c)
    return np.column_stack([pa, np.clip(1.0 - pa - pb, 1e-12, None), pb])


def integrated_probs(d: np.ndarray, var: np.ndarray, beta: float, c: float) -> np.ndarray:
    """Probabilities averaged over a normal posterior on the gap: d + sqrt(var) * z."""
    sd = np.sqrt(var)
    return sum(w * probs(d + sd * x, beta, c) for x, w in zip(GH_X, GH_W))


def fit(
    d: np.ndarray,
    outcome: np.ndarray,
    x0: tuple[float, float] = (0.005, 0.5),
    weights: np.ndarray | None = None,
    var: np.ndarray | None = None,
) -> tuple[float, float]:
    """Maximum likelihood (beta, c). ``outcome`` is 0 = A wins, 1 = draw, 2 = B wins.

    With ``var`` the likelihood integrates over the gap's posterior, so beta describes
    true skill gaps rather than shrunken point estimates.
    """
    d = np.asarray(d, float)
    outcome = np.asarray(outcome)
    w = np.ones(len(d)) if weights is None else np.asarray(weights, float)
    w = w / w.mean()
    scale = max(np.std(d), 1e-9)
    z = d / scale
    zvar = None if var is None else np.asarray(var, float) / scale**2
    rows = np.arange(len(d))

    def nll(theta):
        b, c = theta
        p = probs(z, b, c) if zvar is None else integrated_probs(z, zvar, b, c)
        return -np.sum(w * np.log(np.clip(p[rows, outcome], 1e-12, 1.0)))

    # c >= 0 keeps P(A) + P(B) <= 1; without the bound draw-free data drives c negative.
    res = minimize(nll, np.array([x0[0] * scale, x0[1]]), method="L-BFGS-B", bounds=[(None, None), (0.0, None)])
    return float(res.x[0] / scale), float(res.x[1])
