"""Route a user request to the appropriate optimization model(s).

Strategy:
  - Views  -> Black-Litterman (always).
  - Explicit `primary_goal` string -> that model.
  - Otherwise infer from what user constrained.
  - Always run 1-3 candidates, rank by feasibility + objective.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

from . import models
from .metrics import all_metrics
from .config import RISK_FREE_RATE


PRIMARY_GOALS = {
    "max_sharpe", "max_return", "min_risk", "balanced",
    "min_tail_risk", "min_drawdown", "max_sortino", "max_omega",
    "max_diversification", "black_litterman", "inverse_vol",
}


def _pick_candidates(req) -> list[str]:
    """Return ordered list of model names to try."""
    if req.views:
        return ["black_litterman"]

    if req.primary_goal in PRIMARY_GOALS:
        anchor = req.primary_goal
        # aliases: goals that don't map to a single model directly
        alias = {
            "balanced":     ["risk_parity", "hrp", "max_diversification"],
            "max_return":   ["max_sharpe"],
            # Include max_sharpe on risk-only goals so a strong risk-adjusted
            # alternative can win when it also satisfies the risk cap.
            "min_risk":     ["min_variance", "risk_parity", "max_sharpe"],
            "min_tail_risk": ["min_cvar", "min_variance", "max_sharpe"],
            "min_drawdown": ["min_max_drawdown", "risk_parity", "max_sharpe"],
        }
        if anchor in alias:
            return alias[anchor]
        fallback = {
            "max_sharpe":         ["risk_parity"],
            "max_sortino":        ["max_sharpe"],
            "max_omega":          ["max_sharpe"],
            "max_diversification": ["risk_parity"],
            "inverse_vol":        ["risk_parity"],
        }.get(anchor, [])
        return [anchor] + fallback

    # Constraint-driven inference
    has_ret = req.target_return is not None
    has_vol = req.max_volatility is not None
    has_dd  = req.max_drawdown is not None
    has_cvar = req.max_cvar is not None

    if has_dd:
        return ["min_max_drawdown", "risk_parity", "min_variance"]
    if has_cvar and has_ret:
        return ["min_cvar_for_return", "min_cvar", "min_variance_for_return"]
    if has_cvar:
        return ["min_cvar", "min_variance"]
    if has_ret and has_vol:
        return ["max_return_for_vol", "min_variance_for_return"]
    if has_ret:
        return ["min_variance_for_return", "max_sharpe"]
    if has_vol:
        return ["max_return_for_vol", "min_variance"]
    return ["max_sharpe", "risk_parity", "hrp"]


def _dispatch(name: str, req, mu, cov, returns) -> dict:
    kw = {"w_min": req.w_min, "w_max": req.w_max}
    if name == "max_sharpe":
        return models.max_sharpe(mu, cov, rf=req.risk_free_rate, **kw)
    if name == "min_variance":
        return models.min_variance(mu, cov, **kw)
    if name == "min_variance_for_return":
        return models.min_variance_for_return(mu, cov, req.target_return, **kw)
    if name == "max_return_for_vol":
        return models.max_return_for_vol(mu, cov, req.max_volatility, **kw)
    if name == "min_cvar":
        return models.min_cvar(returns, alpha=req.cvar_alpha, **kw)
    if name == "min_cvar_for_return":
        return models.min_cvar_for_return(returns, mu, req.target_return,
                                          alpha=req.cvar_alpha, **kw)
    if name == "min_max_drawdown":
        return models.min_max_drawdown(returns, target_return=req.target_return, mu=mu, **kw)
    if name == "risk_parity":
        return models.risk_parity(cov, **kw)
    if name == "hrp":
        return models.hierarchical_risk_parity(returns, cov=cov)
    if name == "max_sortino":
        mar_d = (1 + (req.target_return or 0)) ** (1 / 252) - 1
        return models.max_sortino(returns, mar_daily=mar_d, **kw)
    if name == "max_omega":
        mar_d = (1 + (req.target_return or 0)) ** (1 / 252) - 1
        return models.max_omega(returns, mar_daily=mar_d, **kw)
    if name == "max_diversification":
        return models.max_diversification(cov, **kw)
    if name == "inverse_vol":
        return models.inverse_volatility(cov)
    if name == "black_litterman":
        mw = req.market_weights if req.market_weights is not None else _equal_weights(cov)
        return models.black_litterman_weights(
            cov, mw, req.views, tau=req.bl_tau, rf=req.risk_free_rate,
            target_return=req.target_return, **kw,
        )
    raise ValueError(f"Unknown model {name}")


def _equal_weights(cov: pd.DataFrame) -> pd.Series:
    n = len(cov.index)
    return pd.Series(1.0 / n, index=cov.index)


def _feasibility(res: dict, req, mu, cov, returns) -> dict:
    """Check whether the result satisfies user's hard constraints."""
    if res.get("weights") is None:
        return {"feasible": False, "reason": f"solver status={res.get('status')}"}
    w = np.asarray(res["weights"]).flatten()
    if not np.isfinite(w).all() or abs(w.sum() - 1.0) > 1e-3:
        return {"feasible": False, "reason": "weights invalid"}

    violations = []
    if req.w_min > 0.0 and np.any(w < req.w_min - 1e-6):
        violations.append(f"weight below min {req.w_min:.4f}")
    if req.w_max < 1.0 and np.any(w > req.w_max + 1e-6):
        violations.append(f"weight above max {req.w_max:.4f}")

    port = pd.Series(returns.values @ w, index=returns.index)
    m = all_metrics(port, rf=req.risk_free_rate, cvar_alpha=req.cvar_alpha)
    if req.target_return is not None and m["ann_return"] < req.target_return - 1e-4:
        violations.append(f"return {m['ann_return']:.4f} < target {req.target_return:.4f}")
    if req.max_volatility is not None and m["ann_vol"] > req.max_volatility + 1e-4:
        violations.append(f"vol {m['ann_vol']:.4f} > max {req.max_volatility:.4f}")
    if req.max_drawdown is not None and m["max_drawdown"] > req.max_drawdown + 1e-4:
        violations.append(f"maxDD {m['max_drawdown']:.4f} > cap {req.max_drawdown:.4f}")
    if req.max_cvar is not None and m["cvar"] > req.max_cvar + 1e-4:
        violations.append(f"cvar {m['cvar']:.4f} > cap {req.max_cvar:.4f}")
    return {"feasible": not violations, "reason": "; ".join(violations) if violations else "",
            "metrics": m}


def _score(res: dict, feas: dict, req) -> float:
    """Higher = better. Used to rank feasible candidates.

    Risk-only goals (min_risk, min_tail_risk, min_drawdown) get a return-floor
    penalty so a solver that hits the risk metric by loading up negative-return
    assets doesn't beat a slightly-higher-risk solver with a real return.
    """
    if not feas["feasible"]:
        return -1e9 + sum(-1 for _ in feas["reason"].split(";"))
    m = feas["metrics"]
    goal = req.primary_goal or ""
    rf = req.risk_free_rate
    # penalty for returning below the risk-free rate (soft, kicks in when negative-ish)
    ret_penalty = max(0.0, rf - m["ann_return"]) * 0.5
    if goal == "max_return":     return m["ann_return"]
    if goal == "min_risk":       return -m["ann_vol"] - ret_penalty
    if goal == "min_tail_risk":  return -m["cvar"] - ret_penalty * 0.1
    if goal == "min_drawdown":   return -m["max_drawdown"] - ret_penalty * 0.5
    if goal == "max_sortino":    return m["sortino"]
    if goal == "max_omega":      return m["calmar"]
    return m["sharpe"]


def run_selection(req, mu, cov, returns) -> list[dict]:
    """Run all candidate models. Return list sorted by score."""
    candidates = _pick_candidates(req)
    out = []
    for name in candidates:
        try:
            res = _dispatch(name, req, mu, cov, returns)
        except Exception as e:
            out.append({"model": name, "status": "error", "error": str(e),
                        "feasible": False, "score": -1e9})
            continue
        feas = _feasibility(res, req, mu, cov, returns)
        out.append({**res, **feas, "score": _score(res, feas, req)})
    out.sort(key=lambda r: r["score"], reverse=True)
    return out
