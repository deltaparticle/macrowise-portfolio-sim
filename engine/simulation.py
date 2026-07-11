"""Bootstrap simulation for forward-looking outcome distributions.

Optimizers return point estimates (historical mean, volatility). Users
often want distributional answers: "what's the probability this portfolio
achieves 40% total return over 5 years?" This module resamples historical
daily returns with replacement to build an empirical distribution of
horizon-end outcomes, given a fixed set of portfolio weights.

Design choices:
- IID daily bootstrap (not block bootstrap). Simpler and honest about its
  limitation: it does not preserve autocorrelation / vol clustering.
- Non-parametric — no distributional assumption. Fat tails are captured
  as long as they appear in the historical window.
- Weights are held fixed for the entire horizon (no rebalancing). This is
  a static-strategy simulation, matching what /optimize returns.
- Deterministic given a seed.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd

from .config import TRADING_DAYS


@dataclass
class SimulationResult:
    horizon_years: float
    n_simulations: int
    target_total_return: Optional[float]
    total_return: dict[str, float] = field(default_factory=dict)
    annualized_return: dict[str, float] = field(default_factory=dict)
    prob_above_target: Optional[float] = None
    method: str = "iid_bootstrap"
    note: str = ""


def _percentiles(x: np.ndarray) -> dict[str, float]:
    p10, p25, p50, p75, p90 = np.percentile(x, [10, 25, 50, 75, 90])
    return {
        "p10": float(p10),
        "p25": float(p25),
        "median": float(p50),
        "p75": float(p75),
        "p90": float(p90),
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
    }


def bootstrap_simulate(
    weights: dict[str, float],
    returns: pd.DataFrame,
    horizon_years: float = 5.0,
    n_simulations: int = 1000,
    target_total_return: Optional[float] = None,
    seed: int = 42,
) -> SimulationResult:
    """Bootstrap the horizon-end distribution of portfolio total return.

    Parameters
    ----------
    weights : dict[str, float]
        Asset -> weight. Only assets also present in `returns` are used.
        Missing assets are treated as zero weight.
    returns : pd.DataFrame
        Daily returns matrix used to draw scenarios from (rows = dates,
        columns = asset slugs). Typically the same DataFrame the optimizer
        used to compute mu/cov.
    horizon_years : float
        Investment horizon in years. Determines the number of days per
        simulated path: round(horizon_years * TRADING_DAYS).
    n_simulations : int
        Number of paths to draw.
    target_total_return : float, optional
        If provided, report probability that a simulated path exceeded it.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    SimulationResult with percentiles and (if target given) probability.
    """
    horizon_years = float(horizon_years)
    n_simulations = int(n_simulations)
    if horizon_years <= 0:
        raise ValueError("horizon_years must be positive")
    if n_simulations < 1:
        raise ValueError("n_simulations must be >= 1")

    r = returns.dropna(how="all")
    if r.empty:
        raise ValueError("returns DataFrame is empty after dropna")

    # Restrict to assets present in both weights and returns
    w_series = pd.Series(weights, dtype=float).reindex(r.columns).fillna(0.0)
    if float(w_series.sum()) <= 0:
        raise ValueError("weights sum to zero after aligning with returns columns")
    # Renormalize in case the aligned weights don't sum exactly to 1
    w = (w_series / w_series.sum()).values

    horizon_days = max(1, int(round(horizon_years * TRADING_DAYS)))
    R = r.values                                    # (T, N)
    T = R.shape[0]

    port_daily = R @ w                              # (T,) portfolio daily returns
    if not np.all(np.isfinite(port_daily)):
        raise ValueError("non-finite values in portfolio daily returns")

    rng = np.random.default_rng(seed)
    # Draw n_simulations x horizon_days indices with replacement
    idx = rng.integers(0, T, size=(n_simulations, horizon_days))
    sampled = port_daily[idx]                       # (n_sims, horizon_days)

    # Total return per path = prod(1 + r) - 1
    # log-space keeps precision for long horizons
    log1p = np.log1p(sampled)
    total_log = log1p.sum(axis=1)
    total_ret = np.expm1(total_log)                 # (n_sims,)

    # Convert to annualized (CAGR)
    ann_ret = np.expm1(total_log / horizon_years)

    result = SimulationResult(
        horizon_years=horizon_years,
        n_simulations=n_simulations,
        target_total_return=(float(target_total_return)
                             if target_total_return is not None else None),
        total_return=_percentiles(total_ret),
        annualized_return=_percentiles(ann_ret),
        method="iid_bootstrap",
        note=(
            "IID daily bootstrap over the historical lookback window. "
            "Does not preserve autocorrelation or volatility clustering. "
            "Weights held fixed for the entire horizon (no rebalancing). "
            "Distribution reflects historical regimes only — a structurally "
            "different future is not modeled."
        ),
    )
    if target_total_return is not None:
        result.prob_above_target = float(np.mean(total_ret >= target_total_return))
    return result
