"""Charts for the README, from the fits cached by compare.py."""
from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.ticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from regime.data import STRESS, TRAIN_END, load, returns

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
REGIME_MODEL = "MS variance (2 regimes), Student-t"
BENCHMARK = "GARCH(1,1), Student-t"
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def shade_runs(ax, mask: pd.Series, **kw):
    """Shade contiguous runs where mask is True."""
    m = mask.to_numpy()
    edges = np.flatnonzero(np.diff(np.r_[0, m.astype(int), 0]))
    for a, b in zip(edges[::2], edges[1::2]):
        ax.axvspan(mask.index[a], mask.index[b - 1], lw=0, **kw)


def main() -> None:
    with open(ROOT / "data" / "fits.pkl", "rb") as fh:
        cache = pickle.load(fh)
    px = load()
    y = returns(px)
    idx = cache["index"]
    path = cache["paths"][REGIME_MODEL]
    hi = pd.Series(path.filtered[:, -1], idx)
    split = pd.Timestamp(TRAIN_END)

    # ---- 1. regimes over the price history
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True, height_ratios=[1.5, 1])
    spy = px["SPY"].reindex(idx)
    a1.plot(spy.index, spy, color="#1d3557", lw=0.9)
    a1.set_yscale("log")
    a1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    a1.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a1.set_yticks([50, 100, 200, 400, 700])
    shade_runs(a1, hi > 0.5, color="#e63946", alpha=0.18)
    a1.axvline(split, color="0.4", ls="--", lw=0.8)
    a1.text(split, 0.97, "fitted on 2000-2014   ", transform=a1.get_xaxis_transform(), ha="right", va="top", fontsize=8, color="0.3")
    a1.text(split, 0.97, "   out of sample, parameters frozen", transform=a1.get_xaxis_transform(), ha="left", va="top", fontsize=8, color="0.3")
    a1.set_ylabel("SPY (log scale)")
    a1.set_title("Turbulent regime in real time: Markov-switching variance, Student-t (shaded where Pr > 0.5)", loc="left")
    for i, (a, b) in enumerate(STRESS.values()):
        a1.axvspan(pd.Timestamp(a), pd.Timestamp(b), ymin=0, ymax=0.035, color="0.15", lw=0,
                   label="named stress episodes" if i == 0 else None)
    a1.legend(loc="lower right", frameon=False, fontsize=8)
    a2.fill_between(hi.index, hi, color="#e63946", alpha=0.35, lw=0, label="Pr(turbulent | data to date)")
    a2.set_ylim(0, 1)
    a2.set_ylabel("probability")
    ax_v = a2.twinx()
    ax_v.plot(px.index, px["VIX"], color="0.15", lw=0.6, label="VIX (not used by the model)")
    ax_v.set_ylabel("VIX")
    ax_v.spines["top"].set_visible(False)
    h1, l1 = a2.get_legend_handles_labels()
    h2, l2 = ax_v.get_legend_handles_labels()
    a2.legend(h1 + h2, l1 + l2, loc="upper left", frameon=False, fontsize=8)
    a2.xaxis.set_major_locator(mdates.YearLocator(2))
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(DOCS / "regimes.png")
    plt.close(fig)

    # ---- 2. predicted volatility vs VIX, out of sample
    test = idx > split
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for name, color in ((BENCHMARK, "#457b9d"), (REGIME_MODEL, "#e63946")):
        vol = np.sqrt(252 * cache["paths"][name].variance[test])
        ax.plot(idx[test], vol, lw=0.8, color=color, label=f"{name}, one day ahead")
    ax.plot(px.index[px.index > split], px["VIX"][px.index > split], lw=0.7, color="0.2", alpha=0.7, label="VIX")
    ax.set_ylabel("annualised volatility, %")
    ax.set_title("Forecast volatility vs the VIX, 2015 to 2025 (out of sample)", loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(DOCS / "volatility.png")
    plt.close(fig)

    # ---- 3. 1% VaR breaches, out of sample
    fig, ax = plt.subplots(figsize=(10, 3.6))
    yt = y[test]
    ax.bar(yt.index, yt, width=1.5, color="0.35", label="SPY daily return, %")
    for name, color in ((BENCHMARK, "#457b9d"), (REGIME_MODEL, "#e63946")):
        var = pd.Series(cache["var"][(name, 0.01)][test], yt.index)
        breaches = yt < var
        ax.plot(var.index, var, color=color, lw=0.8, label=f"1% VaR, {name} ({breaches.sum()} breaches)")
        ax.scatter(yt.index[breaches], yt[breaches], s=9, color=color, zorder=3)
    ax.set_title("One-day 1% value at risk, 2015 to 2025 (expected breaches: "
                 f"{0.01 * test.sum():.0f})", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.set_ylim(-17, 10)
    fig.tight_layout()
    fig.savefig(DOCS / "var.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
