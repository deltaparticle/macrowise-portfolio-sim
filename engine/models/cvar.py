from __future__ import annotations
import numpy as np
import pandas as pd
import cvxpy as cp

from ._common import base_constraints, solve_cvx, clean_weights


def _cvar_prob(returns: pd.DataFrame, alpha: float, w_min: float, w_max: float,
               extra_cons=None, extra_obj=None):
    """Rockafellar-Uryasev LP for CVaR."""
    assets = list(returns.columns)
    R = returns.values
    T, n = R.shape
    w = cp.Variable(n)
    zeta = cp.Variable()
    u = cp.Variable(T, nonneg=True)
    losses = -R @ w
    cons = base_constraints(w, w_min, w_max) + [u >= losses - zeta]
    if extra_cons:
        cons += extra_cons(w)
    cvar = zeta + cp.sum(u) / (T * alpha)
    obj_expr = cvar if extra_obj is None else extra_obj(w, cvar)
    return assets, w, zeta, u, cvar, cons, obj_expr


def min_cvar(returns: pd.DataFrame, alpha: float = 0.05, w_min=0.0, w_max=1.0):
    assets, w, zeta, u, cvar, cons, obj = _cvar_prob(returns, alpha, w_min, w_max)
    prob = cp.Problem(cp.Minimize(obj), cons)
    status = solve_cvx(prob)
    return {"weights": clean_weights(w.value) if w.value is not None else None,
            "assets": assets, "status": status, "model": "min_cvar",
            "cvar_alpha": alpha, "cvar_daily": float(cvar.value) if cvar.value is not None else None}


def min_cvar_for_return(returns: pd.DataFrame, mu: pd.Series, target_return: float,
                        alpha: float = 0.05, w_min=0.0, w_max=1.0):
    mu_v = mu.reindex(returns.columns).values
    cons_fn = lambda w: [mu_v @ w >= target_return]
    assets, w, zeta, u, cvar, cons, obj = _cvar_prob(returns, alpha, w_min, w_max, extra_cons=cons_fn)
    prob = cp.Problem(cp.Minimize(obj), cons)
    status = solve_cvx(prob)
    return {"weights": clean_weights(w.value) if w.value is not None else None,
            "assets": assets, "status": status, "model": "min_cvar_for_return",
            "cvar_alpha": alpha, "target_return": target_return}
