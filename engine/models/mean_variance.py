from __future__ import annotations
import numpy as np
import pandas as pd
import cvxpy as cp

from ._common import base_constraints, solve_cvx, clean_weights


def _prep(mu: pd.Series, cov: pd.DataFrame):
    assets = list(mu.index)
    n = len(assets)
    Sigma = cov.loc[assets, assets].values
    mu_v = mu.values
    w = cp.Variable(n)
    return assets, n, Sigma, mu_v, w


def min_variance(mu, cov, w_min=0.0, w_max=1.0):
    assets, n, Sigma, _, w = _prep(mu, cov)
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(Sigma))),
                      base_constraints(w, w_min, w_max))
    status = solve_cvx(prob)
    return {"weights": clean_weights(w.value) if w.value is not None else None,
            "assets": assets, "status": status, "model": "min_variance"}


def min_variance_for_return(mu, cov, target_return, w_min=0.0, w_max=1.0):
    assets, n, Sigma, mu_v, w = _prep(mu, cov)
    cons = base_constraints(w, w_min, w_max) + [mu_v @ w >= target_return]
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(Sigma))), cons)
    status = solve_cvx(prob)
    return {"weights": clean_weights(w.value) if w.value is not None else None,
            "assets": assets, "status": status,
            "model": "min_variance_for_return", "target_return": target_return}


def max_return_for_vol(mu, cov, max_vol, w_min=0.0, w_max=1.0):
    assets, n, Sigma, mu_v, w = _prep(mu, cov)
    cons = base_constraints(w, w_min, w_max) + [
        cp.quad_form(w, cp.psd_wrap(Sigma)) <= max_vol ** 2
    ]
    prob = cp.Problem(cp.Maximize(mu_v @ w), cons)
    status = solve_cvx(prob)
    return {"weights": clean_weights(w.value) if w.value is not None else None,
            "assets": assets, "status": status,
            "model": "max_return_for_vol", "max_vol": max_vol}


def max_sharpe(mu, cov, rf=0.065, w_min=0.0, w_max=1.0):
    """Max Sharpe via convex reformulation: min w'Σw s.t. (μ-rf)'w = 1, y>=0.
    Then recover weights = y / sum(y)."""
    assets, n, Sigma, mu_v, _ = _prep(mu, cov)
    excess = mu_v - rf
    if np.all(excess <= 0):
        # no risky asset beats rf → fall back to min variance
        return min_variance(mu, cov, w_min, w_max)
    y = cp.Variable(n, nonneg=True)
    kappa = cp.Variable(nonneg=True)
    cons = [
        excess @ y == 1,
        cp.sum(y) == kappa,
        y >= w_min * kappa,
        y <= w_max * kappa,
    ]
    prob = cp.Problem(cp.Minimize(cp.quad_form(y, cp.psd_wrap(Sigma))), cons)
    status = solve_cvx(prob)
    if y.value is None or kappa.value is None or kappa.value <= 0:
        return {"weights": None, "assets": assets, "status": status, "model": "max_sharpe"}
    w = y.value / kappa.value
    return {"weights": clean_weights(w), "assets": assets, "status": status,
            "model": "max_sharpe"}
