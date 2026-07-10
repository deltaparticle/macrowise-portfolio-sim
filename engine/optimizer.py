"""Main entrypoint. UserRequest -> PortfolioResult."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional
import numpy as np
import pandas as pd

from .config import RISK_FREE_RATE, DEFAULT_LOOKBACK_YEARS
from .data_loader import load_universe
from .sector_map import build_map, indices_for_sectors, market_proxy
from .estimators import historical_mean, james_stein_mean, ledoit_wolf_cov
from .selector import run_selection


@dataclass
class UserRequest:
    # Universe
    sectors: Optional[list[str]] = None            # e.g. ["IT", "Banks"] — from sector_map tags
    universe: Optional[list[str]] = None           # explicit index slugs; overrides sectors
    sector_match_all: bool = False                 # False = any tag match, True = all

    # Targets / constraints (all optional; can combine)
    primary_goal: Optional[str] = None             # one of PRIMARY_GOALS
    target_return: Optional[float] = None          # annualized, e.g. 0.15
    max_volatility: Optional[float] = None         # annualized, e.g. 0.20
    max_drawdown: Optional[float] = None           # e.g. 0.20
    max_cvar: Optional[float] = None               # e.g. 0.03 (daily 5% CVaR)
    cvar_alpha: float = 0.05

    # Weight bounds
    w_min: float = 0.0
    w_max: float = 1.0

    # Data window
    start: Optional[str] = None                    # "YYYY-MM-DD"
    end: Optional[str] = None
    lookback_years: int = DEFAULT_LOOKBACK_YEARS

    # Model knobs
    risk_free_rate: float = RISK_FREE_RATE

    # Black-Litterman
    views: list[dict] = field(default_factory=list)
    market_weights: Optional[pd.Series] = None
    bl_tau: float = 0.05

    # Output
    top_k: int = 3


@dataclass
class PortfolioResult:
    request: dict
    chosen_model: str
    weights: dict
    metrics: dict
    feasible: bool
    reason: str
    all_candidates: list[dict]
    universe: list[str]
    n_assets_used: int


def _resolve_universe(req: UserRequest) -> list[str]:
    if req.universe:
        return list(req.universe)
    tag_map = build_map()
    if req.sectors:
        picks = indices_for_sectors(req.sectors, tag_map, match_any=not req.sector_match_all)
        if not picks and req.sector_match_all:
            picks = indices_for_sectors(req.sectors, tag_map, match_any=True)
        if not picks:
            raise ValueError(f"No indices tagged with sectors={req.sectors}. "
                             f"Try broader tags. Available: see sector_map.available_sectors().")
        return picks
    # Default: use all non-excluded indices
    return [s for s, t in tag_map.items() if not t.exclude]


def _resolve_dates(req: UserRequest, prices: pd.DataFrame) -> tuple[str, str]:
    end = req.end or prices.index.max().strftime("%Y-%m-%d")
    if req.start:
        return req.start, end
    start_ts = pd.Timestamp(end) - pd.DateOffset(years=req.lookback_years)
    start = max(start_ts, prices.index.min()).strftime("%Y-%m-%d")
    return start, end


def optimize(req: UserRequest) -> PortfolioResult:
    universe = _resolve_universe(req)

    prices, returns = load_universe(universe, start=req.start, end=req.end)

    # If date window not given, apply lookback trimming
    if not req.start:
        cutoff = returns.index.max() - pd.DateOffset(years=req.lookback_years)
        returns = returns.loc[returns.index >= cutoff]
        prices = prices.loc[prices.index >= cutoff]

    # Degenerate universes: optimization has one solution.
    if len(returns.columns) == 1:
        asset = returns.columns[0]
        port = returns[asset]
        m = {
            "ann_return": float((1 + port.mean()) ** 252 - 1),
            "ann_vol":    float(port.std(ddof=1) * (252 ** 0.5)),
        }
        m["sharpe"] = (m["ann_return"] - req.risk_free_rate) / m["ann_vol"] if m["ann_vol"] > 0 else 0.0
        # fill remaining metric keys from the full metrics helper
        from .metrics import all_metrics
        m = all_metrics(port, rf=req.risk_free_rate, cvar_alpha=req.cvar_alpha)
        return PortfolioResult(
            request={k: v for k, v in asdict(req).items() if k != "market_weights"},
            chosen_model="single_asset_trivial",
            weights={asset: 1.0},
            metrics=m,
            feasible=True,
            reason="Universe reduced to a single index after history filtering.",
            all_candidates=[],
            universe=[asset],
            n_assets_used=1,
        )

    # Estimation: James-Stein shrunk mean + Ledoit-Wolf shrunk covariance.
    # Both shrink toward stable priors to control MV concentration from
    # noisy sample estimates (Ledoit-Wolf 2003, James-Stein 1961).
    mu = james_stein_mean(returns, annualize=True)
    cov = ledoit_wolf_cov(returns, annualize=True)

    # Market proxy weights for Black-Litterman
    if req.views and req.market_weights is None:
        mp = market_proxy()
        if mp and mp in mu.index:
            mw = pd.Series(0.0, index=mu.index)
            mw[mp] = 1.0
        else:
            # Inverse-vol approximates cap-weighting better than 1/N.
            # Cap-weighted indices tend to over-weight low-vol large-caps;
            # 1/N distorts the equilibrium prior in Black-Litterman.
            vol = np.sqrt(np.diag(cov.values))
            inv = 1.0 / np.where(vol > 0, vol, np.inf)
            mw = pd.Series(inv / inv.sum(), index=mu.index)
        req.market_weights = mw

    candidates = run_selection(req, mu, cov, returns)
    # Prefer candidates that actually produced a portfolio; keep failed ones
    # as fallbacks only if nothing else survived.
    solvable = [c for c in candidates if c.get("weights") is not None and c.get("assets")]
    top = (solvable or candidates)[: req.top_k]

    if not top:
        raise RuntimeError("No candidate models produced a solution.")

    best = top[0]
    assets = best.get("assets", [])
    w = best.get("weights")
    weights_dict = {a: float(x) for a, x in zip(assets, w) if x > 1e-4} if w is not None else {}

    return PortfolioResult(
        request={k: v for k, v in asdict(req).items() if k != "market_weights"},
        chosen_model=best.get("model", "unknown"),
        weights=weights_dict,
        metrics=best.get("metrics", {}),
        feasible=best.get("feasible", False),
        reason=best.get("reason", ""),
        all_candidates=[{
            "model": c.get("model"),
            "status": c.get("status"),
            "feasible": c.get("feasible", False),
            "reason": c.get("reason", ""),
            "score": c.get("score"),
            "metrics": c.get("metrics", {}),
        } for c in top],
        universe=list(mu.index),
        n_assets_used=len(mu.index),
    )
