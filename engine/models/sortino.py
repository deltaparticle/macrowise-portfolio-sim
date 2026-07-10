from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ._common import clean_weights
from ..config import TRADING_DAYS


def max_sortino(returns: pd.DataFrame, mar_daily: float = 0.0,
                w_min: float = 0.0, w_max: float = 1.0):
    """Non-convex. SLSQP with multi-start."""
    assets = list(returns.columns)
    R = returns.values
    n = R.shape[1]

    def neg_sortino(w):
        port = R @ w
        excess = port - mar_daily
        downside = np.minimum(port - mar_daily, 0.0)
        dd_std = np.sqrt(np.mean(downside ** 2))
        if dd_std < 1e-10:
            return -1e6
        return -np.sqrt(TRADING_DAYS) * np.mean(excess) / dd_std

    bounds = [(w_min, w_max)] * n
    cons = [{"type": "eq", "fun": lambda x: np.sum(x) - 1.0}]
    best = None
    for seed in (None, 1, 2):
        rng = np.random.default_rng(seed)
        x0 = np.full(n, 1.0 / n) if seed is None else rng.dirichlet(np.ones(n))
        res = minimize(neg_sortino, x0, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-9})
        if best is None or res.fun < best.fun:
            best = res
    return {"weights": clean_weights(best.x), "assets": assets,
            "status": "optimal" if best.success else "suboptimal",
            "model": "max_sortino", "sortino": float(-best.fun)}
