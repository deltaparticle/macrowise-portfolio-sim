from __future__ import annotations
import numpy as np
import pandas as pd


def inverse_volatility(cov: pd.DataFrame):
    assets = list(cov.index)
    vol = np.sqrt(np.diag(cov.values))
    inv = 1.0 / np.where(vol > 0, vol, np.inf)
    w = inv / inv.sum()
    return {"weights": w, "assets": assets, "status": "optimal",
            "model": "inverse_volatility"}
