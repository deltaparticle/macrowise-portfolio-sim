from __future__ import annotations
import re
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd

from .config import DATA_DIR, MIN_HISTORY_DAYS


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def list_indices(data_dir: Path = DATA_DIR) -> list[str]:
    return sorted(p.stem for p in data_dir.glob("*_yfinance.csv"))


def load_index(name: str, data_dir: Path = DATA_DIR) -> pd.Series:
    path = data_dir / f"{name}.csv" if name.endswith("_yfinance") else data_dir / f"{name}_yfinance.csv"
    if not path.exists():
        path = data_dir / f"{_slug(name)}_yfinance.csv"
    df = pd.read_csv(path, parse_dates=["Date"])
    # normalize timezone: strip any tz info so panels can be concatenated
    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=["Close", "Date"]).sort_values("Date").drop_duplicates("Date")
    s = df.set_index("Date")["Close"].astype(float)
    s.name = path.stem.replace("_yfinance", "")
    return s


def load_price_panel(
    names: Iterable[str],
    start: str | None = None,
    end: str | None = None,
    min_history: int = MIN_HISTORY_DAYS,
) -> pd.DataFrame:
    series = []
    for n in names:
        try:
            s = load_index(n)
        except FileNotFoundError:
            continue
        if start:
            s = s[s.index >= pd.Timestamp(start)]
        if end:
            s = s[s.index <= pd.Timestamp(end)]
        if len(s) >= min_history:
            series.append(s)
    if not series:
        raise ValueError("No indices with sufficient history in the given range.")
    px = pd.concat(series, axis=1).sort_index()
    px = px.dropna(how="all")
    return px


def compute_returns(prices: pd.DataFrame, log: bool = False) -> pd.DataFrame:
    if log:
        r = np.log(prices / prices.shift(1))
    else:
        r = prices.pct_change()
    return r.dropna(how="all")


def align_returns(prices: pd.DataFrame, min_overlap: int = MIN_HISTORY_DAYS) -> pd.DataFrame:
    """Restrict to date range where >= 90% of indices have data, drop cols failing min_overlap."""
    coverage = prices.notna().mean(axis=1)
    mask = coverage >= 0.90
    if mask.sum() < min_overlap:
        # fall back: use full range but forward-fill up to 5 days
        pass
    else:
        prices = prices.loc[mask]
    prices = prices.ffill(limit=5).dropna(axis=1, thresh=min_overlap)
    prices = prices.dropna(how="any")
    return prices


def load_universe(
    names: list[str],
    start: str | None = None,
    end: str | None = None,
    log_returns: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    px = load_price_panel(names, start=start, end=end)
    px = align_returns(px)
    rets = compute_returns(px, log=log_returns)
    return px, rets
