"""Fit every model on 2000-2014, then score them one day ahead on 2015 to September 2025.

Writes the tables in docs/ and caches fitted paths in data/ for report.py.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from regime.data import STRESS, TRAIN_END, load, returns
from regime.evaluate import diebold_mariano, qlike, var_backtest
from regime.models import Spec, fit

ROOT = Path(__file__).resolve().parent.parent
DOCS, DATA = ROOT / "docs", ROOT / "data"

SPECS = [
    Spec(1, True, False),
    Spec(1, True, True),
    Spec(2, False, False),
    Spec(2, False, True),
    Spec(2, True, False),
    Spec(2, True, True),
    Spec(2, True, True, min_stay=0.95),
]
BENCHMARK = Spec(1, True, True).name


def main() -> None:
    px = load()
    y = returns(px)
    train, test = y.index <= TRAIN_END, y.index > TRAIN_END
    fits, paths = {}, {}
    for spec in SPECS:
        f = fit(y[train].to_numpy(), spec)
        fits[spec.name], paths[spec.name] = f, f.run(y.to_numpy())
        print(f"fitted {spec.name}: loglik {f.loglik:.1f}")

    # ---- in-sample fit
    rows = []
    for name, f in fits.items():
        p, path = f.params, paths[name]
        in_path = f.run(y[train].to_numpy())
        rows.append({
            "model": name, "params": f.spec.n_params, "log-likelihood": f.loglik, "AIC": f.aic, "BIC": f.bic,
            "regime vol % (calm / turbulent)": " / ".join(f"{v:.0f}" for v in f.regime_vol(in_path)),
            "regime duration days": " / ".join(f"{d:.0f}" for d in f.expected_duration()) if f.spec.regimes > 1 else "",
        "stay probabilities": " / ".join(f"{q:.3f}" for q in np.diag(p["P"])) if f.spec.regimes > 1 else "",
            "nu": p["nu"],
        })
    pd.DataFrame(rows).to_csv(DOCS / "in_sample.csv", index=False, float_format="%.3f")

    # ---- out-of-sample density and variance forecasts
    yt = y[test].to_numpy()
    base_score = -paths[BENCHMARK].log_score[test]
    base_qlike = qlike(paths[BENCHMARK].variance[test], yt - fits[BENCHMARK].params["mu"])
    rows = []
    for name, path in paths.items():
        score = -path.log_score[test]
        ql = qlike(path.variance[test], yt - fits[name].params["mu"])
        dm_s = diebold_mariano(score, base_score) if name != BENCHMARK else (np.nan, np.nan)
        dm_q = diebold_mariano(ql, base_qlike) if name != BENCHMARK else (np.nan, np.nan)
        rows.append({"model": name, "avg log score": -score.mean(), "log score DM vs GARCH-t": dm_s[0],
                     "DM p": dm_s[1], "QLIKE": ql.mean(), "QLIKE DM vs GARCH-t": dm_q[0], "QLIKE DM p": dm_q[1]})
    pd.DataFrame(rows).to_csv(DOCS / "out_of_sample.csv", index=False, float_format="%.4f")

    # ---- value-at-risk backtests
    rows, var_paths = [], {}
    for name, f in fits.items():
        for level in (0.01, 0.05):
            var = f.value_at_risk(paths[name], level)
            var_paths[(name, level)] = var
            rows.append({"model": name, "level": level, **var_backtest(yt, var[test], level)})
    pd.DataFrame(rows).to_csv(DOCS / "var_backtest.csv", index=False, float_format="%.4f")

    # ---- do the turbulent regimes line up with known stress episodes?
    vix = px["VIX"].reindex(y.index)
    rows = []
    for name, path in paths.items():
        if fits[name].spec.regimes == 1:
            continue
        hi = pd.Series(path.filtered[:, -1], y.index)
        row = {"model": name}
        for episode, (a, b) in STRESS.items():
            row[episode] = (hi[a:b] > 0.5).mean()
        calm = np.ones(len(y), bool)
        for a, b in STRESS.values():
            calm &= ~((y.index >= a) & (y.index <= b))
        row["outside episodes"] = (hi[calm] > 0.5).mean()
        row["outside episodes (test)"] = (hi[calm & test] > 0.5).mean()
        row["corr with VIX"] = hi.corr(vix)
        row["corr with VIX (test)"] = hi[test].corr(vix[test])
        rows.append(row)
    pd.DataFrame(rows).to_csv(DOCS / "stress.csv", index=False, float_format="%.3f")

    params = {name: {k: (np.asarray(v).tolist() if v is not None else None) for k, v in f.params.items()}
              for name, f in fits.items()}
    (DOCS / "params.json").write_text(json.dumps(params, indent=2))
    with open(DATA / "fits.pkl", "wb") as fh:
        pickle.dump({"fits": fits, "paths": paths, "var": var_paths, "index": y.index}, fh)


if __name__ == "__main__":
    main()
