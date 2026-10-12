"""GARCH(1,1), Markov-switching variance and Markov-switching GARCH, fitted by maximum likelihood.

Returns are y_t = mu + eps_t with eps_t = sigma_{s_t, t} z_t, z_t a unit-variance Gaussian or
Student-t innovation, and s_t a hidden Markov chain. Three nested specifications:

* GARCH(1,1)        one regime:  sigma2_t = omega + alpha eps2_{t-1} + beta sigma2_{t-1}
* MS variance       K regimes, constant variance sigma2_k in each (Hamilton 1989)
* MS-GARCH          K regimes, each with its own GARCH(1,1) recursion (Haas, Mittnik and
                    Paolella 2004). Every regime's variance is updated with the observed eps,
                    not with its own hypothetical past, so sigma2_{k,t} is known at t - 1 and
                    the Hamilton filter computes the likelihood exactly. The Gray (1996) and
                    Klaassen (2002) variants instead need an approximation to avoid a likelihood
                    that depends on the whole regime path.

Regimes are labelled from calm to turbulent by their average variance over the fitting sample.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.special import expit, logit

from . import dists
from .markov import hamilton_filter, kim_smoother, stationary


@dataclass(frozen=True)
class Spec:
    regimes: int = 1
    garch: bool = True
    student_t: bool = False

    @property
    def name(self) -> str:
        core = ("GARCH(1,1)" if self.garch else "Constant variance") if self.regimes == 1 else (
            f"MS-GARCH ({self.regimes} regimes)" if self.garch else f"MS variance ({self.regimes} regimes)")
        return f"{core}, {'Student-t' if self.student_t else 'Gaussian'}"

    @property
    def n_params(self) -> int:
        k = self.regimes
        return 1 + k * (3 if self.garch else 1) + int(self.student_t) + k * (k - 1)


def garch_variance(eps: np.ndarray, omega: float, alpha: float, beta: float, s0: float) -> np.ndarray:
    """sigma2_0 = s0, then sigma2_t = omega + alpha eps_{t-1}^2 + beta sigma2_{t-1}.

    Returns T + 1 values: the last is the forecast for the day after the sample.
    """
    x = omega + alpha * np.asarray(eps, float) ** 2
    rest = lfilter([1.0], [1.0, -beta], x, zi=[beta * s0])[0]
    return np.r_[s0, rest]


# ---------------------------------------------------------------- parameter packing
def _unpack(spec: Spec, theta: np.ndarray) -> dict:
    """Unconstrained vector -> parameters: omega > 0, alpha + beta < 1, nu > 2, valid P."""
    k, i = spec.regimes, 1
    out = {"mu": theta[0], "omega": [], "alpha": [], "beta": []}
    for _ in range(k):
        out["omega"].append(np.exp(theta[i]))
        if spec.garch:
            persistence, share = 0.9995 * expit(theta[i + 1]), expit(theta[i + 2])
            out["alpha"].append(persistence * share)
            out["beta"].append(persistence * (1 - share))
            i += 3
        else:
            out["alpha"].append(0.0)
            out["beta"].append(0.0)
            i += 1
    out["nu"] = 2.05 + np.exp(theta[i]) if spec.student_t else None
    i += int(spec.student_t)
    P = np.ones((1, 1))
    if k > 1:
        # Each row: K - 1 free logits against a zero for the diagonal, then softmax.
        P = np.empty((k, k))
        for r in range(k):
            z = np.zeros(k)
            z[[c for c in range(k) if c != r]] = theta[i : i + k - 1]
            i += k - 1
            P[r] = np.exp(z - z.max()) / np.exp(z - z.max()).sum()
    out["P"] = P
    return out


def _pack(spec: Spec, p: dict) -> np.ndarray:
    k, theta = spec.regimes, [p["mu"]]
    for j in range(k):
        theta.append(np.log(p["omega"][j]))
        if spec.garch:
            s = p["alpha"][j] + p["beta"][j]
            theta += [logit(s / 0.9995), logit(p["alpha"][j] / s)]
    if spec.student_t:
        theta.append(np.log(p["nu"] - 2.05))
    for r in range(k if k > 1 else 0):
        theta += [np.log(p["P"][r, c] / p["P"][r, r]) for c in range(k) if c != r]
    return np.array(theta, float)


# ---------------------------------------------------------------- likelihood
@dataclass
class Path:
    """Everything the model says about a return series, one step ahead and in hindsight."""
    sigma2: np.ndarray          # T x K regime variances known at t - 1
    predicted: np.ndarray       # Pr(s_t = k | info to t - 1)
    filtered: np.ndarray        # Pr(s_t = k | info to t), real time
    smoothed: np.ndarray        # Pr(s_t = k | all data), hindsight
    log_score: np.ndarray       # one-step predictive log density of y_t
    loglik: float

    @property
    def variance(self) -> np.ndarray:
        """One-step predictive variance of y_t (all regimes share mu)."""
        return (self.predicted * self.sigma2).sum(axis=1)


def _regime_variances(spec: Spec, p: dict, eps: np.ndarray, s0: float) -> np.ndarray:
    k = spec.regimes
    if not spec.garch:
        return np.tile(np.array(p["omega"]), (len(eps), 1))
    cols = [garch_variance(eps[:-1], p["omega"][j], p["alpha"][j], p["beta"][j], s0) for j in range(k)]
    return np.column_stack(cols)


def _loglik(spec: Spec, theta: np.ndarray, y: np.ndarray, s0: float) -> float:
    p = _unpack(spec, theta)
    eps = y - p["mu"]
    s2 = _regime_variances(spec, p, eps, s0)
    log_dens = dists.logpdf(eps[:, None], s2, p["nu"])
    if spec.regimes == 1:
        return float(log_dens.sum())
    return hamilton_filter(log_dens, p["P"]).loglik


# ---------------------------------------------------------------- fitted model
@dataclass
class Fit:
    spec: Spec
    params: dict
    loglik: float
    n_obs: int
    s0: float                                   # initial variance, from the fitting sample
    theta: np.ndarray = field(repr=False)

    @property
    def aic(self) -> float:
        return 2 * self.spec.n_params - 2 * self.loglik

    @property
    def bic(self) -> float:
        return np.log(self.n_obs) * self.spec.n_params - 2 * self.loglik

    def regime_vol(self, path: "Path") -> np.ndarray:
        """Average annualised volatility of each regime along a filtered path (percent)."""
        return np.sqrt(252 * path.sigma2.mean(axis=0))

    def expected_duration(self) -> np.ndarray:
        """Average days spent in each regime per visit: 1 / (1 - p_kk)."""
        return 1 / np.maximum(1 - np.diag(self.params["P"]), 1e-12)

    def run(self, y: np.ndarray) -> Path:
        """Filter (and smooth) a series with these fixed parameters."""
        y = np.asarray(y, float)
        p = self.params
        eps = y - p["mu"]
        s2 = _regime_variances(self.spec, p, eps, self.s0)
        log_dens = dists.logpdf(eps[:, None], s2, p["nu"])
        fr = hamilton_filter(log_dens, p["P"])
        return Path(s2, fr.predicted, fr.filtered, kim_smoother(fr, p["P"]), fr.log_score, fr.loglik)

    def value_at_risk(self, path: Path, level: float = 0.01) -> np.ndarray:
        """One-day VaR as a return quantile (negative = loss) of the predictive mixture.

        Solved by bisection: F(q) = sum_k Pr(s_t = k) F_k(q) is increasing in q.
        """
        mu, nu = self.params["mu"], self.params["nu"]
        sd = np.sqrt(path.sigma2.max(axis=1))
        lo, hi = mu - 30 * sd, mu + 30 * sd
        for _ in range(80):
            mid = (lo + hi) / 2
            cdf = (path.predicted * dists.cdf((mid - mu)[:, None], path.sigma2, nu)).sum(axis=1)
            below = cdf < level
            lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
        return (lo + hi) / 2


def _canonical_order(spec: Spec, p: dict, y: np.ndarray, s0: float) -> dict:
    """Relabel regimes from calm to turbulent by their average variance over the sample.

    The average path is used rather than omega / (1 - alpha - beta), which explodes when a
    regime's GARCH is close to integrated even if its variance never gets large in practice.
    """
    if spec.regimes == 1:
        return p
    order = np.argsort(_regime_variances(spec, p, y - p["mu"], s0).mean(axis=0))
    q = dict(p)
    for key in ("omega", "alpha", "beta"):
        q[key] = [p[key][j] for j in order]
    q["P"] = p["P"][np.ix_(order, order)]
    return q


def _starts(spec: Spec, y: np.ndarray, rng: np.random.Generator, n: int) -> list[np.ndarray]:
    v, k = y.var(), spec.regimes
    base = {"mu": y.mean(), "nu": 8.0}
    if spec.garch:
        scale = np.geomspace(0.5, 2.5, k) if k > 1 else np.ones(1)
        base.update(omega=list(0.02 * v * scale), alpha=[0.08] * k, beta=[0.90] * k)
    else:
        base.update(omega=list(v * np.geomspace(0.4, 3.0, k)), alpha=[0.0] * k, beta=[0.0] * k)
    stay = 0.98
    base["P"] = np.full((k, k), (1 - stay) / max(k - 1, 1)) + np.eye(k) * (stay - (1 - stay) / max(k - 1, 1))
    theta0 = _pack(spec, base)
    return [theta0] + [theta0 + rng.normal(0, 0.6, theta0.size) for _ in range(n - 1)]


def fit(y, spec: Spec, n_starts: int = 6, seed: int = 0) -> Fit:
    """Maximum likelihood over several starting points; the best local optimum wins."""
    y = np.asarray(y, float)
    s0 = float(y.var())
    rng = np.random.default_rng(seed)
    best = None
    for theta0 in _starts(spec, y, rng, n_starts):
        obj = lambda th: -_loglik(spec, th, y, s0)
        try:
            res = minimize(obj, theta0, method="L-BFGS-B")
            res = minimize(obj, res.x, method="Nelder-Mead",
                           options={"maxiter": 4000, "xatol": 1e-7, "fatol": 1e-9})
        except (FloatingPointError, ValueError):
            continue
        if np.isfinite(res.fun) and (best is None or res.fun < best.fun):
            best = res
    if best is None:
        raise RuntimeError(f"{spec.name}: every start failed")
    params = _canonical_order(spec, _unpack(spec, best.x), y, s0)
    return Fit(spec, params, -float(best.fun), len(y), s0, _pack(spec, params))


def simulate(spec: Spec, params: dict, n: int, seed: int = 0):
    """Simulate (returns, regimes) from the model; used by the tests."""
    rng = np.random.default_rng(seed)
    k, P = spec.regimes, np.asarray(params["P"]) if spec.regimes > 1 else np.ones((1, 1))
    omega, alpha, beta = (np.asarray(params[x], float) for x in ("omega", "alpha", "beta"))
    s2 = omega / (1 - alpha - beta) if spec.garch else omega.copy()
    z = dists.sample(rng, n, params.get("nu"))
    s = rng.choice(k, p=stationary(P))
    y, states = np.empty(n), np.empty(n, int)
    for t in range(n):
        if t:
            s = rng.choice(k, p=P[s])
        states[t] = s
        eps = np.sqrt(s2[s]) * z[t]
        y[t] = params["mu"] + eps
        if spec.garch:
            s2 = omega + alpha * eps**2 + beta * s2
    return y, states
