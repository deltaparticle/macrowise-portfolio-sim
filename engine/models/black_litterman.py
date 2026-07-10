from __future__ import annotations
import numpy as np
import pandas as pd

from ..estimators import implied_equilibrium_returns
from .mean_variance import max_sharpe, min_variance_for_return


def _build_PQ(views: list[dict], assets: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Views: list of dicts. Two shapes:
      absolute:  {"asset": "nifty_it", "return": 0.15, "confidence": 0.6}
      relative:  {"long": ["nifty_it"], "short": ["nifty_pharma"], "return": 0.03, "confidence": 0.5}
    Confidence in (0, 1]. Higher = tighter Omega.
    """
    n = len(assets)
    idx = {a: i for i, a in enumerate(assets)}
    P_rows, Q, conf = [], [], []
    for v in views:
        row = np.zeros(n)
        if "asset" in v:
            if v["asset"] not in idx:
                continue
            row[idx[v["asset"]]] = 1.0
        else:
            longs = [a for a in v.get("long", []) if a in idx]
            shorts = [a for a in v.get("short", []) if a in idx]
            if not longs:
                continue
            for a in longs:
                row[idx[a]] = 1.0 / len(longs)
            for a in shorts:
                row[idx[a]] = -1.0 / max(len(shorts), 1)
        P_rows.append(row)
        Q.append(float(v["return"]))
        conf.append(float(v.get("confidence", 0.5)))
    if not P_rows:
        return np.zeros((0, n)), np.zeros(0), np.zeros(0)
    return np.vstack(P_rows), np.array(Q), np.array(conf)


def _omega_from_confidence(P: np.ndarray, cov_arr: np.ndarray, tau: float,
                            confidence: np.ndarray) -> np.ndarray:
    """Idzorek-lite: Omega_ii = (1/c_i - 1) * tau * p_i' Sigma p_i, clipped."""
    conf = np.clip(confidence, 1e-3, 1 - 1e-3)
    diag = []
    for i, p in enumerate(P):
        v = tau * float(p @ cov_arr @ p)
        diag.append((1.0 / conf[i] - 1.0) * v)
    return np.diag(diag)


def black_litterman_posterior(
    cov: pd.DataFrame,
    market_weights: pd.Series,
    views: list[dict],
    tau: float = 0.05,
    risk_aversion: float | None = None,
    market_excess_return: float = 0.06,
) -> tuple[pd.Series, pd.DataFrame]:
    """Return (posterior mean, posterior covariance)."""
    assets = list(cov.index)
    pi = implied_equilibrium_returns(cov, market_weights,
                                     risk_aversion=risk_aversion,
                                     market_excess_return=market_excess_return).values
    Sigma = cov.values
    P, Q, conf = _build_PQ(views, assets)
    if P.shape[0] == 0:
        return pd.Series(pi, index=assets), cov.copy()
    Omega = _omega_from_confidence(P, Sigma, tau, conf)
    tauS = tau * Sigma
    A = np.linalg.inv(tauS) + P.T @ np.linalg.inv(Omega) @ P
    b = np.linalg.inv(tauS) @ pi + P.T @ np.linalg.inv(Omega) @ Q
    mu_bl = np.linalg.solve(A, b)
    M_inv = np.linalg.inv(A)
    cov_bl = Sigma + M_inv
    return pd.Series(mu_bl, index=assets), pd.DataFrame(cov_bl, index=assets, columns=assets)


def black_litterman_weights(
    cov: pd.DataFrame,
    market_weights: pd.Series,
    views: list[dict],
    tau: float = 0.05,
    rf: float = 0.065,
    target_return: float | None = None,
    w_min: float = 0.0, w_max: float = 1.0,
):
    mu_bl, cov_bl = black_litterman_posterior(cov, market_weights, views, tau=tau)
    if target_return is not None:
        res = min_variance_for_return(mu_bl, cov_bl, target_return, w_min, w_max)
    else:
        res = max_sharpe(mu_bl, cov_bl, rf=rf, w_min=w_min, w_max=w_max)
    res["model"] = "black_litterman"
    res["posterior_mu"] = mu_bl.to_dict()
    return res
