import numpy as np
from scipy import stats

from regime.evaluate import christoffersen, diebold_mariano, kupiec, qlike


def test_kupiec_accepts_the_exact_rate_and_rejects_double():
    hits = np.zeros(1000, bool)
    hits[:10] = True
    assert np.isclose(kupiec(hits, 0.01)[0], 0.0) and np.isclose(kupiec(hits, 0.01)[1], 1.0)
    hits[:20] = True
    lr, p = kupiec(hits, 0.01)
    # Closed form for x = 20 breaches in n = 1000 at p = 1%
    expected = -2 * (980 * np.log(0.99) + 20 * np.log(0.01) - 980 * np.log(0.98) - 20 * np.log(0.02))
    assert np.isclose(lr, expected) and p < 0.01


def test_christoffersen_flags_clustered_breaches_only():
    spread = np.zeros(1000, int)
    spread[::50] = 1
    clustered = np.zeros(1000, int)
    clustered[100:120] = 1
    assert christoffersen(spread)[1] > 0.1
    assert christoffersen(clustered)[1] < 1e-6


def test_qlike_is_minimised_by_the_true_variance():
    r = np.random.default_rng(0).normal(0, 2.0, 200_000)
    losses = {h: qlike(np.full(r.size, h), r).mean() for h in (2.0, 3.0, 4.0, 5.0, 6.0)}
    assert min(losses, key=losses.get) == 4.0


def test_diebold_mariano_detects_a_better_forecast():
    rng = np.random.default_rng(1)
    common = rng.normal(size=5000)
    stat, p = diebold_mariano(common, common + 0.05 + rng.normal(0, 0.2, 5000))
    assert stat < 0 and p < 0.01
    stat, p = diebold_mariano(common + rng.normal(0, 0.2, 5000), common + rng.normal(0, 0.2, 5000))
    assert p > 0.01
    # Under the null with iid losses the statistic is approximately standard normal
    assert abs(stat) < stats.norm.ppf(0.995)
