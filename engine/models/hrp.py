from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


def _correl_dist(corr: pd.DataFrame) -> pd.DataFrame:
    return np.sqrt(0.5 * (1 - corr))


def _quasi_diag(link: np.ndarray) -> list[int]:
    link = link.astype(int)
    sortIx = pd.Series([link[-1, 0], link[-1, 1]])
    numItems = link[-1, 3]
    while sortIx.max() >= numItems:
        sortIx.index = range(0, sortIx.shape[0] * 2, 2)
        df0 = sortIx[sortIx >= numItems]
        i = df0.index
        j = df0.values - numItems
        sortIx[i] = link[j, 0]
        df0 = pd.Series(link[j, 1], index=i + 1)
        sortIx = pd.concat([sortIx, df0]).sort_index()
        sortIx.index = range(sortIx.shape[0])
    return sortIx.tolist()


def _ivp(cov: pd.DataFrame, idx: list[int]) -> np.ndarray:
    v = np.diag(cov.values)[idx]
    ivp = 1.0 / v
    return ivp / ivp.sum()


def _cluster_var(cov: pd.DataFrame, idx: list[int]) -> float:
    sub = cov.iloc[idx, idx].values
    w = _ivp(cov, idx).reshape(-1, 1)
    return float((w.T @ sub @ w).item())


def _rec_bisect(cov: pd.DataFrame, sortIx: list[int]) -> pd.Series:
    w = pd.Series(1.0, index=sortIx)
    clusters = [sortIx]
    while clusters:
        clusters = [c[j:k] for c in clusters
                    for j, k in ((0, len(c) // 2), (len(c) // 2, len(c)))
                    if len(c) > 1]
        for i in range(0, len(clusters), 2):
            c0, c1 = clusters[i], clusters[i + 1]
            v0 = _cluster_var(cov, c0)
            v1 = _cluster_var(cov, c1)
            alpha = 1 - v0 / (v0 + v1)
            w[c0] *= alpha
            w[c1] *= 1 - alpha
    return w


def hierarchical_risk_parity(returns: pd.DataFrame, cov: pd.DataFrame | None = None):
    """López de Prado HRP."""
    assets = list(returns.columns)
    if cov is None:
        cov = returns.cov()
    corr = returns.corr()
    dist = _correl_dist(corr)
    link = linkage(squareform(dist.values, checks=False), method="single")
    sortIx = _quasi_diag(link)
    w = _rec_bisect(cov.reset_index(drop=True), sortIx)
    w = w.reindex(range(len(assets))).values
    w = w / w.sum()
    return {"weights": w, "assets": assets, "status": "optimal", "model": "hrp"}
