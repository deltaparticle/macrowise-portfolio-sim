from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ._common import clean_weights


def max_diversification(cov: pd.DataFrame, w_min: float = 0.0, w_max: float = 1.0):
    """Choueifaty diversification ratio: (w' * sigma) / sqrt(w' Sigma w). Max it."""
    assets = list(cov.index)
    Sigma = cov.values
    sigma = np.sqrt(np.diag(Sigma))
    n = len(assets)

    def neg_dr(w):
        num = float(w @ sigma)
        den = float(np.sqrt(w @ Sigma @ w))
        return -num / max(den, 1e-12)

    x0 = np.full(n, 1.0 / n)
    bounds = [(w_min, w_max)] * n
    cons = [{"type": "eq", "fun": lambda x: np.sum(x) - 1.0}]
    res = minimize(neg_dr, x0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 300, "ftol": 1e-9})
    return {"weights": clean_weights(res.x), "assets": assets,
            "status": "optimal" if res.success else "suboptimal",
            "model": "max_diversification",
            "diversification_ratio": float(-res.fun)}
