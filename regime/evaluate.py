"""Out-of-sample scoring: density forecasts, variance forecasts and value-at-risk backtests."""
from __future__ import annotations

import numpy as np
from scipy import stats


def qlike(variance_forecast, returns) -> np.ndarray:
    """QLIKE loss log h + r^2 / h, robust to the noise in squared returns (Patton 2011)."""
    h = np.asarray(variance_forecast, float)
    return np.log(h) + np.asarray(returns, float) ** 2 / h


def diebold_mariano(loss_a, loss_b, lags: int = 5) -> tuple[float, float]:
    """DM statistic for mean(loss_a - loss_b) with a Newey-West variance, and two-sided p-value.

    Negative statistics mean model a has the lower loss.
    """
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    n, dc = len(d), d - d.mean()
    lrv = dc @ dc / n
    for lag in range(1, lags + 1):
        lrv += 2 * (1 - lag / (lags + 1)) * (dc[lag:] @ dc[:-lag]) / n
    stat = d.mean() / np.sqrt(lrv / n)
    return float(stat), float(2 * stats.norm.sf(abs(stat)))


def kupiec(hits, level: float) -> tuple[float, float]:
    """Kupiec (1995) proportion-of-failures test: LR statistic and p-value (chi-square, 1 df)."""
    hits = np.asarray(hits, bool)
    n, x = len(hits), int(hits.sum())
    phat = x / n

    def ll(p):
        return (n - x) * np.log1p(-p) + (x * np.log(p) if x else 0.0)

    lr = -2 * (ll(level) - (ll(phat) if 0 < x < n else (0.0 if x == 0 else -np.inf)))
    return float(lr), float(stats.chi2.sf(lr, 1))


def christoffersen(hits) -> tuple[float, float]:
    """Christoffersen (1998) independence test: do VaR breaches cluster? LR and p-value."""
    h = np.asarray(hits, int)
    prev, cur = h[:-1], h[1:]
    n = {(i, j): int(((prev == i) & (cur == j)).sum()) for i in (0, 1) for j in (0, 1)}

    def term(count, p):
        return count * np.log(p) if count else 0.0

    p01 = n[0, 1] / max(n[0, 0] + n[0, 1], 1)
    p11 = n[1, 1] / max(n[1, 0] + n[1, 1], 1)
    p = (n[0, 1] + n[1, 1]) / max(sum(n.values()), 1)
    alt = term(n[0, 0], 1 - p01) + term(n[0, 1], p01) + term(n[1, 0], 1 - p11) + term(n[1, 1], p11)
    null = term(n[0, 0] + n[1, 0], 1 - p) + term(n[0, 1] + n[1, 1], p)
    lr = max(-2 * (null - alt), 0.0)
    return float(lr), float(stats.chi2.sf(lr, 1))


def var_backtest(returns, var, level: float) -> dict:
    hits = np.asarray(returns) < np.asarray(var)
    return {
        "breaches": int(hits.sum()),
        "expected": level * len(hits),
        "rate": hits.mean(),
        "kupiec p": kupiec(hits, level)[1],
        "independence p": christoffersen(hits)[1],
    }
