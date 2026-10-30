"""Three-way regulation outcome model on a rating gap d.

P(A wins) = sigma(beta*d - c), P(B wins) = sigma(-beta*d - c), draw = the rest.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, log_expit


def probs(d: np.ndarray, beta: float, c: float) -> np.ndarray:
    pa = expit(beta * d - c)
    pb = expit(-beta * d - c)
    return np.column_stack([pa, np.clip(1.0 - pa - pb, 1e-12, None), pb])


def fit(d: np.ndarray, outcome: np.ndarray, x0: tuple[float, float] = (0.005, 0.5)) -> tuple[float, float]:
    """Maximum likelihood (beta, c). ``outcome`` is 0 = A wins, 1 = draw, 2 = B wins."""
    d = np.asarray(d, float)
    outcome = np.asarray(outcome)
    scale = max(np.std(d), 1e-9)
    z = d / scale

    def nll(theta):
        b, c = theta
        la = log_expit(b * z - c)
        lb = log_expit(-b * z - c)
        pd_ = np.clip(1.0 - np.exp(la) - np.exp(lb), 1e-12, None)
        ll = np.where(outcome == 0, la, np.where(outcome == 2, lb, np.log(pd_)))
        return -ll.sum()

    # c >= 0 keeps P(A) + P(B) <= 1; without the bound draw-free data drives c negative.
    res = minimize(nll, np.array([x0[0] * scale, x0[1]]), method="L-BFGS-B", bounds=[(None, None), (0.0, None)])
    return float(res.x[0] / scale), float(res.x[1])
