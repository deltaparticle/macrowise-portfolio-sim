from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize

from ._common import clean_weights


def _portfolio_dd(w: np.ndarray, R: np.ndarray) -> float:
    port_ret = R @ w
    curve = np.cumprod(1 + port_ret)
    peak = np.maximum.accumulate(curve)
    dd = (curve - peak) / peak
    return float(-dd.min())


def _simplex_project(w: np.ndarray, w_min: float, w_max: float) -> np.ndarray:
    w = np.clip(w, w_min, w_max)
    s = w.sum()
    if s <= 0:
        w = np.full_like(w, 1.0 / len(w))
    else:
        w = w / s
    return np.clip(w, w_min, w_max)


def min_max_drawdown(returns: pd.DataFrame, w_min: float = 0.0, w_max: float = 1.0,
                     target_return: float | None = None,
                     mu: pd.Series | None = None,
                     seed: int = 42, maxiter: int = 5):
    """Non-convex — differential evolution over the simplex.
    If target_return supplied, penalize returns below it."""
    assets = list(returns.columns)
    R = returns.values
    n = R.shape[1]
    mu_v = mu.reindex(assets).values if mu is not None else R.mean(axis=0) * 252

    def obj(x):
        w = _simplex_project(x, w_min, w_max)
        dd = _portfolio_dd(w, R)
        if target_return is not None:
            r = float(mu_v @ w)
            if r < target_return:
                dd += 10 * (target_return - r)  # penalty
        return dd

    bounds = [(w_min, w_max)] * n
    result = differential_evolution(
        obj, bounds, seed=seed, maxiter=maxiter,
        popsize=15, tol=2e-3, polish=True, workers=1,
        init="sobol",
    )
    w = _simplex_project(result.x, w_min, w_max)
    return {"weights": clean_weights(w), "assets": assets,
            "status": "optimal" if result.success else "suboptimal",
            "model": "min_max_drawdown", "max_drawdown": float(result.fun)}
