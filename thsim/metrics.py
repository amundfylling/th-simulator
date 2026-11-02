"""Scoring rules for probabilistic forecasts."""

from __future__ import annotations

import numpy as np

EPS = 1e-15


def log_loss(p: np.ndarray, outcome: np.ndarray) -> float:
    p = np.asarray(p, float)
    if p.ndim == 1:  # binary: p = P(event), outcome in {0, 1}
        p = np.column_stack([1.0 - p, p])
    return float(-np.mean(np.log(np.clip(p[np.arange(len(p)), outcome], EPS, 1.0))))


def brier(p: np.ndarray, outcome: np.ndarray) -> float:
    p = np.asarray(p, float)
    if p.ndim == 1:
        return float(np.mean((p - outcome) ** 2))
    onehot = np.eye(p.shape[1])[outcome]
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def rps(p: np.ndarray, outcome: np.ndarray) -> float:
    """Ranked probability score for ordered categories."""
    p = np.asarray(p, float)
    onehot = np.eye(p.shape[1])[outcome]
    cum = np.cumsum(p, axis=1)[:, :-1] - np.cumsum(onehot, axis=1)[:, :-1]
    return float(np.mean(np.sum(cum**2, axis=1) / (p.shape[1] - 1)))


def reliability(p: np.ndarray, hit: np.ndarray, bins: int = 10) -> list[tuple[float, float, int]]:
    """(mean forecast, observed frequency, count) per equal-width probability bin."""
    p = np.asarray(p, float)
    hit = np.asarray(hit, float)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    out = []
    for b in range(bins):
        m = idx == b
        if m.any():
            out.append((float(p[m].mean()), float(hit[m].mean()), int(m.sum())))
    return out
