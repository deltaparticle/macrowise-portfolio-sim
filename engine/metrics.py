from __future__ import annotations
import numpy as np
import pandas as pd

from .config import TRADING_DAYS, RISK_FREE_RATE


def portfolio_returns(weights: np.ndarray, returns: pd.DataFrame) -> pd.Series:
    return pd.Series(returns.values @ np.asarray(weights).flatten(), index=returns.index)


def ann_return(daily: pd.Series) -> float:
    return float((1 + daily.mean()) ** TRADING_DAYS - 1)


def ann_vol(daily: pd.Series) -> float:
    return float(daily.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe(daily: pd.Series, rf: float = RISK_FREE_RATE) -> float:
    v = ann_vol(daily)
    return (ann_return(daily) - rf) / v if v > 0 else 0.0


def sortino(daily: pd.Series, rf: float = RISK_FREE_RATE) -> float:
    rf_daily = (1 + rf) ** (1 / TRADING_DAYS) - 1
    downside = np.minimum(daily - rf_daily, 0)
    dd = np.sqrt(np.mean(downside ** 2)) * np.sqrt(TRADING_DAYS)
    return (ann_return(daily) - rf) / dd if dd > 0 else 0.0


def max_drawdown(daily: pd.Series) -> float:
    curve = (1 + daily).cumprod()
    peak = curve.cummax()
    dd = (curve - peak) / peak
    return float(-dd.min())


def cvar(daily: pd.Series, alpha: float = 0.05) -> float:
    losses = -daily
    var = np.quantile(losses, 1 - alpha)
    tail = losses[losses >= var]
    return float(tail.mean()) if len(tail) else float(var)


def calmar(daily: pd.Series) -> float:
    dd = max_drawdown(daily)
    return ann_return(daily) / dd if dd > 0 else 0.0


def all_metrics(daily: pd.Series, rf: float = RISK_FREE_RATE, cvar_alpha: float = 0.05) -> dict:
    return {
        "ann_return":   ann_return(daily),
        "ann_vol":      ann_vol(daily),
        "sharpe":       sharpe(daily, rf),
        "sortino":      sortino(daily, rf),
        "max_drawdown": max_drawdown(daily),
        "cvar":         cvar(daily, cvar_alpha),
        "calmar":       calmar(daily),
    }
