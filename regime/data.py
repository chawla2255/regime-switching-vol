"""S&P 500 (SPY) daily returns and the VIX index, downloaded once and cached locally."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

START, END = "2000-01-01", "2025-09-30"
TRAIN_END = "2014-12-31"            # parameters are fitted on 2000-2014 only
CACHE = Path(__file__).resolve().parent.parent / "data" / "spy_vix.csv.gz"

# Stress episodes used to check the regimes after the fact. The models never see these dates.
STRESS = {
    "Global financial crisis": ("2008-09-15", "2009-03-31"),
    "US downgrade": ("2011-08-01", "2011-10-31"),
    "China devaluation": ("2015-08-17", "2015-09-30"),
    "Volmageddon": ("2018-02-02", "2018-02-28"),
    "Q4 2018 selloff": ("2018-10-03", "2018-12-31"),
    "COVID crash": ("2020-02-21", "2020-04-30"),
    "2022 bear market": ("2022-01-03", "2022-10-31"),
    "Tariff shock": ("2025-04-02", "2025-04-30"),
}


def load(refresh: bool = False) -> pd.DataFrame:
    """Daily SPY adjusted close and VIX close."""
    if refresh or not CACHE.exists():
        import yfinance as yf

        raw = yf.download(["SPY", "^VIX"], start=START, end=END, auto_adjust=True, progress=False)
        px = raw["Close"].rename(columns={"^VIX": "VIX"})[["SPY", "VIX"]].dropna()
        CACHE.parent.mkdir(exist_ok=True)
        px.to_csv(CACHE)
    return pd.read_csv(CACHE, index_col=0, parse_dates=True)


def returns(px: pd.DataFrame) -> pd.Series:
    """Daily SPY log returns in percent (the scale GARCH optimisers like)."""
    return (100 * np.log(px["SPY"]).diff()).dropna().rename("ret")
