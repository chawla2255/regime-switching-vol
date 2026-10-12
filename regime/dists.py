"""Innovation densities with zero mean and unit variance, scaled by a conditional volatility.

Student-t is standardised (scaled by sqrt((nu - 2) / nu)) so that sigma is the conditional
standard deviation for every nu > 2, which keeps GARCH variances comparable across densities.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.special import gammaln


def logpdf(eps, sigma2, nu: float | None = None) -> np.ndarray:
    """Log density of eps given conditional variance sigma2; Gaussian if nu is None."""
    eps, sigma2 = np.asarray(eps, float), np.asarray(sigma2, float)
    if nu is None:
        return -0.5 * (np.log(2 * np.pi) + np.log(sigma2) + eps**2 / sigma2)
    const = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2))
    return const - 0.5 * np.log(sigma2) - (nu + 1) / 2 * np.log1p(eps**2 / (sigma2 * (nu - 2)))


def cdf(x, sigma2, nu: float | None = None) -> np.ndarray:
    z = np.asarray(x, float) / np.sqrt(sigma2)
    if nu is None:
        return stats.norm.cdf(z)
    return stats.t.cdf(z / np.sqrt((nu - 2) / nu), nu)


def ppf(q, nu: float | None = None) -> np.ndarray:
    """Quantile of the unit-variance innovation."""
    if nu is None:
        return stats.norm.ppf(q)
    return stats.t.ppf(q, nu) * np.sqrt((nu - 2) / nu)


def sample(rng: np.random.Generator, size, nu: float | None = None) -> np.ndarray:
    if nu is None:
        return rng.standard_normal(size)
    return rng.standard_t(nu, size) * np.sqrt((nu - 2) / nu)
