from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from .config import TRADING_DAYS, RISK_FREE_RATE


def annualize_return(daily_mean: pd.Series, periods: int = TRADING_DAYS) -> pd.Series:
    return (1 + daily_mean) ** periods - 1


def annualize_cov(daily_cov: pd.DataFrame, periods: int = TRADING_DAYS) -> pd.DataFrame:
    return daily_cov * periods


def historical_mean(returns: pd.DataFrame, annualize: bool = True) -> pd.Series:
    mu = returns.mean()
    return annualize_return(mu) if annualize else mu


def ewma_mean(returns: pd.DataFrame, halflife: int = 63, annualize: bool = True) -> pd.Series:
    mu = returns.ewm(halflife=halflife).mean().iloc[-1]
    return annualize_return(mu) if annualize else mu


def james_stein_mean(returns: pd.DataFrame, annualize: bool = True) -> pd.Series:
    """Positive-part James-Stein shrinkage of sample mean toward grand mean.

    Sample means are noisy; JS shrinkage pulls them toward the cross-sectional
    average, dominating the sample mean in MSE for n >= 3. Standard cure for
    the mean-variance concentration problem driven by mean estimation error.
    """
    r = returns.dropna(how="any")
    T, N = r.shape
    if N < 3 or T < 10:
        return historical_mean(returns, annualize=annualize)
    mu_hat = r.mean().values
    grand = float(mu_hat.mean())
    # per-asset sample variance of the mean estimator = sigma^2 / T
    sigma2 = r.var().values / T
    dev = mu_hat - grand
    ss = float(np.sum(dev ** 2))
    if ss <= 1e-16:
        shrunk = mu_hat
    else:
        shrink = 1.0 - (N - 2) * float(np.mean(sigma2)) / ss
        shrink = max(0.0, min(1.0, 1.0 - shrink))  # positive-part, clamp to [0,1]
        shrunk = shrink * grand + (1 - shrink) * mu_hat
    mu = pd.Series(shrunk, index=r.columns)
    return annualize_return(mu) if annualize else mu


def sample_cov(returns: pd.DataFrame, annualize: bool = True) -> pd.DataFrame:
    c = returns.cov()
    return annualize_cov(c) if annualize else c


def ledoit_wolf_cov(returns: pd.DataFrame, annualize: bool = True) -> pd.DataFrame:
    """Ledoit-Wolf shrinkage — industry default for stable optimization."""
    r = returns.dropna()
    lw = LedoitWolf().fit(r.values)
    cov = pd.DataFrame(lw.covariance_, index=r.columns, columns=r.columns)
    return annualize_cov(cov) if annualize else cov


def ewma_cov(returns: pd.DataFrame, halflife: int = 63, annualize: bool = True) -> pd.DataFrame:
    r = returns.dropna()
    c = r.ewm(halflife=halflife).cov(pairwise=True).groupby(level=1).last()
    c = c.loc[r.columns, r.columns]
    return annualize_cov(c) if annualize else c


def vol_regime_ratio(returns: pd.DataFrame, short_window: int = 30) -> float:
    """Cross-sectional median of (recent short-window vol / full-window vol).
    >1 means current volatility is elevated vs the full lookback.
    Used to auto-switch between Ledoit-Wolf (calm) and EWMA (stressed) cov."""
    r = returns.dropna()
    if len(r) < short_window * 2:
        return 1.0
    recent = r.tail(short_window)
    short_vol = recent.std(ddof=1)
    long_vol = r.std(ddof=1)
    ratio = (short_vol / long_vol.replace(0, np.nan)).dropna()
    if ratio.empty:
        return 1.0
    return float(ratio.median())


def adaptive_cov(
    returns: pd.DataFrame,
    threshold: float = 1.3,
    short_window: int = 30,
    ewma_halflife: int = 63,
    annualize: bool = True,
) -> tuple[pd.DataFrame, str, float]:
    """Regime-aware covariance. Returns (cov, method_used, vol_ratio).

    Rationale: Ledoit-Wolf averages across the full lookback, so during a
    volatility regime shift it lags reality (underestimates current risk).
    EWMA weights recent observations more heavily, tracking the current
    regime. We switch when recent vol is materially above long-run vol.
    """
    ratio = vol_regime_ratio(returns, short_window=short_window)
    if ratio >= threshold:
        return ewma_cov(returns, halflife=ewma_halflife, annualize=annualize), "ewma", ratio
    return ledoit_wolf_cov(returns, annualize=annualize), "ledoit_wolf", ratio


def implied_equilibrium_returns(
    cov: pd.DataFrame,
    market_weights: pd.Series,
    risk_aversion: float | None = None,
    market_excess_return: float = 0.06,
    market_variance: float | None = None,
) -> pd.Series:
    """Reverse-optimization: Pi = delta * Sigma * w_mkt (Black-Litterman prior)."""
    w = market_weights.reindex(cov.index).fillna(0.0).values
    if risk_aversion is None:
        var_m = market_variance if market_variance is not None else float(w @ cov.values @ w)
        risk_aversion = market_excess_return / max(var_m, 1e-8)
    pi = risk_aversion * cov.values @ w
    return pd.Series(pi, index=cov.index)


def portfolio_stats(
    weights: np.ndarray,
    mu: pd.Series,
    cov: pd.DataFrame,
    rf: float = RISK_FREE_RATE,
) -> dict:
    w = np.asarray(weights).flatten()
    ret = float(w @ mu.values)
    vol = float(np.sqrt(w @ cov.values @ w))
    sharpe = (ret - rf) / vol if vol > 0 else 0.0
    return {"expected_return": ret, "volatility": vol, "sharpe": sharpe}
