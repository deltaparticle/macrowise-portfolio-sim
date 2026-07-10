from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ._common import clean_weights


def risk_parity(cov: pd.DataFrame, w_min: float = 1e-4, w_max: float = 1.0):
    """Equal Risk Contribution — Spinu / Bruder-Roncalli SLSQP solver."""
    assets = list(cov.index)
    Sigma = cov.values
    n = len(assets)

    def obj(w):
        w = w / w.sum()
        port_vol = np.sqrt(w @ Sigma @ w)
        mrc = Sigma @ w / max(port_vol, 1e-12)
        rc = w * mrc
        target = port_vol / n
        return float(np.sum((rc - target) ** 2))

    x0 = np.full(n, 1.0 / n)
    bounds = [(w_min, w_max)] * n
    cons = [{"type": "eq", "fun": lambda x: np.sum(x) - 1.0}]
    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-10})
    return {"weights": clean_weights(res.x), "assets": assets,
            "status": "optimal" if res.success else "suboptimal",
            "model": "risk_parity"}
