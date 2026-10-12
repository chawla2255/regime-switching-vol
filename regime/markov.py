"""Hidden Markov chain machinery: the Hamilton (1989) filter and the Kim (1994) smoother.

Both take the matrix of log densities log f(y_t | s_t = k), so they work for any model whose
regime-k density at time t depends only on past observations, not on the path of regimes.
That is what makes the Haas et al. (2004) MS-GARCH below exactly tractable.

Transition matrix convention: P[i, j] = Pr(s_t = j | s_{t-1} = i), rows sum to one.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def stationary(P: np.ndarray) -> np.ndarray:
    """Ergodic distribution pi with pi P = pi."""
    vals, vecs = np.linalg.eig(P.T)
    v = np.real(vecs[:, np.argmin(np.abs(vals - 1))])
    return v / v.sum()


@dataclass
class FilterResult:
    loglik: float
    predicted: np.ndarray     # Pr(s_t = k | y_1..y_{t-1}), T x K
    filtered: np.ndarray      # Pr(s_t = k | y_1..y_t), T x K
    log_score: np.ndarray     # log f(y_t | y_1..y_{t-1}), one-step predictive log density


def hamilton_filter(log_dens: np.ndarray, P: np.ndarray, init: np.ndarray | None = None) -> FilterResult:
    """`init` is the distribution of the first regime; the chain's ergodic one by default."""
    log_dens = np.asarray(log_dens, float)
    t_len, k = log_dens.shape
    xi = stationary(P) if init is None else np.asarray(init, float)
    predicted, filtered, score = np.empty((t_len, k)), np.empty((t_len, k)), np.empty(t_len)
    peak = log_dens.max(axis=1)
    dens = np.exp(log_dens - peak[:, None])          # rescaled per row to avoid underflow
    for t in range(t_len):
        pred = xi if t == 0 else xi @ P
        joint = pred * dens[t]
        c = joint.sum()
        xi = joint / c
        predicted[t], filtered[t], score[t] = pred, xi, np.log(c) + peak[t]
    return FilterResult(float(score.sum()), predicted, filtered, score)


def kim_smoother(fr: FilterResult, P: np.ndarray) -> np.ndarray:
    """Pr(s_t = k | y_1..y_T), T x K."""
    smoothed = np.empty_like(fr.filtered)
    smoothed[-1] = fr.filtered[-1]
    for t in range(len(smoothed) - 2, -1, -1):
        ratio = smoothed[t + 1] / np.maximum(fr.predicted[t + 1], 1e-300)
        s = fr.filtered[t] * (P @ ratio)
        smoothed[t] = s / s.sum()
    return smoothed
