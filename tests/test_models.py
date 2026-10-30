import numpy as np

from thsim import metrics
from thsim.models import ordered_logit


def test_ordered_logit_recovers_parameters():
    rng = np.random.default_rng(0)
    d = rng.normal(0, 250, 200_000)
    p = ordered_logit.probs(d, 0.006, 0.4)
    y = (rng.random(len(d))[:, None] > np.cumsum(p, axis=1)).sum(axis=1)
    beta, c = ordered_logit.fit(d, y)
    assert abs(beta - 0.006) < 3e-4 and abs(c - 0.4) < 0.03


def test_metrics_on_known_values():
    p = np.array([[0.5, 0.25, 0.25], [0.2, 0.3, 0.5]])
    y = np.array([0, 2])
    assert np.isclose(metrics.log_loss(p, y), -(np.log(0.5) + np.log(0.5)) / 2)
    assert np.isclose(metrics.brier(p, y), ((0.25 + 0.0625 + 0.0625) + (0.04 + 0.09 + 0.25)) / 2)
    # RPS: cum diffs (-0.5, -0.25) and (0.2, 0.5) over K-1 = 2
    assert np.isclose(metrics.rps(p, y), ((0.25 + 0.0625) / 2 + (0.04 + 0.25) / 2) / 2)
