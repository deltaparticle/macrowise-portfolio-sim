"""Walk-forward backtest: periodically re-optimize on rolling window."""
from __future__ import annotations
from dataclasses import replace
import numpy as np
import pandas as pd

from .optimizer import UserRequest, optimize
from .data_loader import load_universe
from .sector_map import build_map, indices_for_sectors
from .metrics import all_metrics


def _rebalance_dates(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    freq = freq.upper()
    if freq in ("M", "MONTHLY"):    return index.to_series().resample("MS").first().dropna().index
    if freq in ("Q", "QUARTERLY"):  return index.to_series().resample("QS").first().dropna().index
    if freq in ("A", "Y", "ANNUAL"): return index.to_series().resample("AS").first().dropna().index
    if freq in ("W", "WEEKLY"):     return index.to_series().resample("W").first().dropna().index
    raise ValueError(f"Unknown freq {freq}")


def walk_forward(
    req: UserRequest,
    rebalance: str = "Q",
    lookback_years: int = 3,
    initial_capital: float = 100.0,
) -> dict:
    """Roll through history, re-run optimizer at each rebalance, track equity curve."""
    universe = req.universe or [
        s for s, t in build_map().items()
        if not t.exclude and (not req.sectors or set(req.sectors) & set(t.sectors + t.themes + t.sizes + t.styles))
    ]
    prices, returns = load_universe(universe, start=None, end=req.end)
    dates = _rebalance_dates(returns.index, rebalance)

    equity = pd.Series(index=returns.index, dtype=float)
    equity.iloc[0] = initial_capital
    w = None
    weights_log = []

    for i, d in enumerate(dates):
        window_start = d - pd.DateOffset(years=lookback_years)
        if window_start < returns.index.min():
            continue
        sub_req = replace(req, start=window_start.strftime("%Y-%m-%d"),
                          end=d.strftime("%Y-%m-%d"),
                          lookback_years=lookback_years)
        try:
            res = optimize(sub_req)
            w = pd.Series(res.weights).reindex(returns.columns).fillna(0.0)
            weights_log.append({"date": d, "model": res.chosen_model,
                                "weights": res.weights})
        except Exception as e:
            weights_log.append({"date": d, "error": str(e)})
            if w is None:
                continue

        next_d = dates[i + 1] if i + 1 < len(dates) else returns.index.max()
        segment = returns.loc[(returns.index > d) & (returns.index <= next_d)]
        port_ret = segment @ w.reindex(segment.columns).fillna(0.0).values
        curve_start = equity.loc[:d].dropna().iloc[-1] if equity.loc[:d].notna().any() else initial_capital
        curve = curve_start * (1 + port_ret).cumprod()
        equity.loc[curve.index] = curve

    equity = equity.dropna().ffill()
    daily = equity.pct_change().dropna()
    return {
        "equity_curve": equity,
        "daily_returns": daily,
        "metrics": all_metrics(daily, rf=req.risk_free_rate, cvar_alpha=req.cvar_alpha),
        "rebalance_log": weights_log,
        "rebalance_dates": list(dates),
    }
