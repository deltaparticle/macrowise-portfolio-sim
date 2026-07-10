from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ._common import clean_weights


def max_omega(returns: pd.DataFrame, mar_daily: float = 0.0,
              w_min: float = 0.0, w_max: float = 1.0):
    """Omega = E[max(r - MAR, 0)] / E[max(MAR - r, 0)]. SLSQP with multi-start."""
    assets = list(returns.columns)
    R = returns.values
    n = R.shape[1]

    def neg_omega(w):
        port = R @ w
        gains = np.maximum(port - mar_daily, 0.0).mean()
        losses = np.maximum(mar_daily - port, 0.0).mean()
        if losses < 1e-12:
            return -1e6
        return -gains / losses

    bounds = [(w_min, w_max)] * n
    cons = [{"type": "eq", "fun": lambda x: np.sum(x) - 1.0}]
    best = None
    for seed in (None, 1, 2):
        rng = np.random.default_rng(seed)
        x0 = np.full(n, 1.0 / n) if seed is None else rng.dirichlet(np.ones(n))
        res = minimize(neg_omega, x0, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-9})
        if best is None or res.fun < best.fun:
            best = res
    return {"weights": clean_weights(best.x), "assets": assets,
            "status": "optimal" if best.success else "suboptimal",
            "model": "max_omega", "omega": float(-best.fun)}
