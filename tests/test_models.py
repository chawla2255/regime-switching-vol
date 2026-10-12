import numpy as np

from regime.markov import hamilton_filter
from regime.models import Spec, _loglik, _pack, _unpack, fit, garch_variance, simulate


def test_garch_recursion_matches_explicit_loop():
    eps = np.random.default_rng(0).normal(size=300)
    s2 = garch_variance(eps, 0.05, 0.08, 0.9, 1.3)
    ref = [1.3]
    for e in eps:
        ref.append(0.05 + 0.08 * e**2 + 0.9 * ref[-1])
    assert np.allclose(s2, ref)


def test_pack_and_unpack_round_trip():
    spec = Spec(regimes=3, garch=True, student_t=True)
    p = {"mu": 0.03, "omega": [0.01, 0.05, 0.2], "alpha": [0.03, 0.08, 0.15], "beta": [0.95, 0.88, 0.7],
         "nu": 6.0, "P": np.array([[0.97, 0.02, 0.01], [0.05, 0.9, 0.05], [0.02, 0.18, 0.8]])}
    q = _unpack(spec, _pack(spec, p))
    for key in ("mu", "omega", "alpha", "beta", "nu"):
        assert np.allclose(q[key], p[key])
    assert np.allclose(q["P"], p["P"])
    assert spec.n_params == len(_pack(spec, p))


def test_identical_regimes_collapse_to_single_garch():
    """With the same GARCH in both regimes the regime chain is irrelevant to the likelihood."""
    y = np.random.default_rng(2).normal(0.02, 1.1, 800)
    one = {"mu": 0.02, "omega": [0.04], "alpha": [0.07], "beta": [0.9], "nu": 7.0, "P": np.ones((1, 1))}
    two = dict(one, omega=[0.04] * 2, alpha=[0.07] * 2, beta=[0.9] * 2, P=np.array([[0.9, 0.1], [0.3, 0.7]]))
    ll1 = _loglik(Spec(1, True, True), _pack(Spec(1, True, True), one), y, 1.2)
    ll2 = _loglik(Spec(2, True, True), _pack(Spec(2, True, True), two), y, 1.2)
    assert np.isclose(ll1, ll2)


def test_garch_fit_recovers_simulated_parameters():
    spec = Spec(1, True, True)
    true = {"mu": 0.05, "omega": [0.03], "alpha": [0.09], "beta": [0.89], "nu": 6.0}
    y, _ = simulate(spec, true, 20_000, seed=3)
    f = fit(y, spec, n_starts=2)
    p = f.params
    assert abs(p["alpha"][0] - 0.09) < 0.015 and abs(p["beta"][0] - 0.89) < 0.02
    assert abs(p["nu"] - 6.0) < 1.0 and abs(p["mu"] - 0.05) < 0.02


def test_switching_variance_fit_recovers_regimes_and_labels_them():
    spec = Spec(2, False, False)
    true = {"mu": 0.0, "omega": [4.0, 0.5], "alpha": [0, 0], "beta": [0, 0],
            "P": np.array([[0.95, 0.05], [0.02, 0.98]])}
    y, states = simulate(spec, true, 6000, seed=4)
    f = fit(y, spec, n_starts=2)
    p = f.params
    # Labels come back ordered calm -> turbulent whatever the simulation order was
    assert np.allclose(p["omega"], [0.5, 4.0], rtol=0.12)
    assert np.allclose(np.diag(p["P"]), [0.98, 0.95], atol=0.015)
    smoothed = f.run(y).smoothed[:, 1]
    assert ((smoothed > 0.5) == (states == 0)).mean() > 0.95


def test_var_is_the_quantile_of_the_predictive_mixture():
    spec = Spec(2, True, True)
    p = {"mu": 0.02, "omega": [0.01, 0.1], "alpha": [0.03, 0.1], "beta": [0.95, 0.85], "nu": 5.0,
         "P": np.array([[0.98, 0.02], [0.04, 0.96]])}
    y, _ = simulate(spec, p, 3000, seed=5)
    from regime.models import Fit
    f = Fit(spec, p, 0.0, len(y), float(y.var()), _pack(spec, p))
    path = f.run(y)
    var = f.value_at_risk(path, 0.05)
    from regime import dists
    cdf = (path.predicted * dists.cdf((var - p["mu"])[:, None], path.sigma2, p["nu"])).sum(axis=1)
    assert np.allclose(cdf, 0.05, atol=1e-9)
    # Run's one-step log score adds up to the filter likelihood
    assert np.isclose(path.log_score.sum(), path.loglik)
