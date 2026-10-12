import itertools

import numpy as np

from regime.markov import hamilton_filter, kim_smoother, stationary


def brute_force(log_dens, P, init):
    """Enumerate every regime path: likelihood and exact posterior marginals."""
    t_len, k = log_dens.shape
    marg, total = np.zeros((t_len, k)), 0.0
    for path in itertools.product(range(k), repeat=t_len):
        p = init[path[0]] * np.exp(log_dens[0, path[0]])
        for t in range(1, t_len):
            p *= P[path[t - 1], path[t]] * np.exp(log_dens[t, path[t]])
        total += p
        marg[np.arange(t_len), path] += p
    return np.log(total), marg / total


def _case(seed, t_len=7, k=2):
    rng = np.random.default_rng(seed)
    log_dens = rng.normal(-1.0, 1.5, (t_len, k))
    P = rng.dirichlet(np.ones(k) * 2, size=k)
    return log_dens, P


def test_stationary_distribution_is_invariant():
    _, P = _case(0, k=3)
    pi = stationary(P)
    assert np.allclose(pi @ P, pi) and np.isclose(pi.sum(), 1)


def test_filter_likelihood_matches_path_enumeration():
    for seed in range(5):
        log_dens, P = _case(seed, k=2 + seed % 2)
        fr = hamilton_filter(log_dens, P)
        ll, _ = brute_force(log_dens, P, stationary(P))
        assert np.isclose(fr.loglik, ll, atol=1e-10)
        assert np.allclose(fr.filtered.sum(axis=1), 1) and np.allclose(fr.predicted.sum(axis=1), 1)


def test_smoother_matches_exact_posterior():
    for seed in range(5):
        log_dens, P = _case(seed + 10, k=2 + seed % 2)
        fr = hamilton_filter(log_dens, P)
        _, exact = brute_force(log_dens, P, stationary(P))
        assert np.allclose(kim_smoother(fr, P), exact, atol=1e-10)


def test_filter_survives_extreme_log_densities():
    log_dens, P = _case(3, t_len=50)
    log_dens = log_dens - 2000.0          # densities of 1e-870 would underflow without rescaling
    fr = hamilton_filter(log_dens, P)
    assert np.isfinite(fr.loglik) and np.allclose(fr.filtered.sum(axis=1), 1)
