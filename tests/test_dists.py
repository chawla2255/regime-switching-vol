import numpy as np
from scipy.integrate import quad

from regime import dists


def test_densities_integrate_to_one_with_unit_variance():
    for nu in (None, 4.5, 10.0):
        f = lambda x: np.exp(dists.logpdf(x, 1.0, nu))
        assert np.isclose(quad(f, -np.inf, np.inf)[0], 1, atol=1e-8)
        assert np.isclose(quad(lambda x: x * x * f(x), -np.inf, np.inf)[0], 1, atol=1e-6)


def test_quantile_inverts_cdf():
    for nu in (None, 5.0):
        q = dists.ppf(np.array([0.01, 0.05, 0.5]), nu)
        assert np.allclose(dists.cdf(q * 2.0, 4.0, nu), [0.01, 0.05, 0.5])
