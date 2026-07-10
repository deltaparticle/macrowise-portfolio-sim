#!/usr/bin/env python3
"""
Master downloader for every index in indian_indices_list.csv.

Sources (all official & free):
  - NSE indices (145): niftyindices.com  POST /BackPage/getHistoricaldatatabletoString
  - BSE indices (125): api.bseindia.com   GET  /BseIndiaAPI/api/ProduceCSVForDatePAR/w
  - India VIX pre-2018: Yahoo Finance     ^INDIAVIX  (VIX from niftyindices caps at 2018;
                                                       yfinance goes back to launch 2008-03)

Each file is written to  data/<slug>_yfinance.csv  (kept as "_yfinance" suffix
purely for backward compatibility with earlier notebooks in this repo).

Usage:
    python collect_indices.py                 # download missing indices only
    python collect_indices.py --force         # re-download everything
    python collect_indices.py --only nse      # nse|bse|vix|report
    python collect_indices.py --report        # only regenerate data collection.md
    python collect_indices.py --start 2000-01-01  --end 2026-07-10

Dependencies:  pip install requests pandas yfinance
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from io import StringIO
from typing import Iterable

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------------------------------------------------------------------
# Config / paths
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CSV_PATH = os.path.join(BASE_DIR, "indian_indices_list.xlsx - indian_indices_list.csv")

DEFAULT_START = datetime(2000, 1, 1)
DEFAULT_END = datetime.now()

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

os.makedirs(DATA_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(BASE_DIR, "collect_indices.log"), encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("collect")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def slugify(name: str) -> str:
    s = name.lower().replace("&", "and")
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


def target_path(name: str) -> str:
    return os.path.join(DATA_DIR, f"{slugify(name)}_yfinance.csv")


def read_indices() -> list[dict]:
    with open(CSV_PATH, encoding="utf-8") as f:
        return [
            {"name": r["short_name"].strip(), "exchange": r["exchange"].strip()}
            for r in csv.DictReader(f)
            if r["short_name"].strip()
        ]


def build_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=4,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"]),
    )
    s.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20))
    return s


def add_return_and_vol(df: pd.DataFrame) -> pd.DataFrame:
    if "Close" in df.columns and not df.empty:
        df["Return"] = df["Close"].pct_change()
        df["Volatility"] = df["Return"].rolling(252).std() * (252 ** 0.5)
    return df


# ---------------------------------------------------------------------------
# NSE — niftyindices.com
# ---------------------------------------------------------------------------

NIFTY_URL = "https://www.niftyindices.com/BackPage/getHistoricaldatatabletoString"
NIFTY_WARMUP = "https://www.niftyindices.com/reports/historical-data"
NIFTY_HEADERS = {
    "Content-Type": "application/json; charset=UTF-8",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "User-Agent": USER_AGENT,
    "Referer": NIFTY_WARMUP,
    "Origin": "https://www.niftyindices.com",
    "X-Requested-With": "XMLHttpRequest",
}


def _nifty_date(d: datetime) -> str:
    return d.strftime("%d-%b-%Y")


def fetch_nifty(session: requests.Session, index_name: str,
                start: datetime, end: datetime) -> pd.DataFrame | None:
    api_name = index_name.upper()
    payload = {
        "cinfo": json.dumps({
            "name": api_name,
            "startDate": _nifty_date(start),
            "endDate": _nifty_date(end),
            "indexName": api_name,
        })
    }
    try:
        r = session.post(NIFTY_URL, headers=NIFTY_HEADERS, json=payload, timeout=45)
        r.raise_for_status()
    except requests.RequestException as e:
        log.warning("NSE %s: HTTP error %s", index_name, e)
        return None

    body = r.text.strip()
    if not body or body[:1] not in "[{":
        log.warning("NSE %s: non-JSON response", index_name)
        return None
    try:
        data = json.loads(body)
        if isinstance(data, dict) and "d" in data:
            inner = data["d"]
            records = json.loads(inner) if isinstance(inner, str) else inner
        else:
            records = data
    except ValueError as e:
        log.warning("NSE %s: parse error %s", index_name, e)
        return None
    if not records:
        return None

    df = pd.DataFrame(records)
    col_map = {
        "HistoricalDate": "Date", "Date": "Date",
        "OPEN": "Open", "Open Index Value": "Open",
        "HIGH": "High", "High Index Value": "High",
        "LOW": "Low",  "Low Index Value": "Low",
        "CLOSE": "Close", "Closing Index Value": "Close",
        "TotalTradedVolume": "Volume", "Volume": "Volume",
    }
    df = df.rename(columns={c: col_map[c] for c in df.columns if c in col_map})
    for noise in ("RequestNumber", "Index Name", "INDEX_NAME"):
        if noise in df.columns:
            df = df.drop(columns=noise)
    if "Date" not in df.columns:
        return None
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["Date"]).sort_values("Date").set_index("Date")
    for c in ("Open", "High", "Low", "Close", "Volume"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "", regex=False), errors="coerce")
    return add_return_and_vol(df)


# ---------------------------------------------------------------------------
# BSE — api.bseindia.com
# ---------------------------------------------------------------------------

BSE_DROPDOWN_URL = "https://api.bseindia.com/BseIndiaAPI/api/FillddlIndex/w?fmdt=&todt="
BSE_CSV_URL      = "https://api.bseindia.com/BseIndiaAPI/api/ProduceCSVForDatePAR/w"
BSE_HEADERS = {
    "User-Agent": USER_AGENT,
    "Referer": "https://www.bseindia.com/",
    "Origin": "https://www.bseindia.com",
    "Accept": "application/json, text/plain, */*",
}


def _norm_name(s: str) -> str:
    s = s.upper().replace("&", "AND").replace("(INR)", "")
    s = s.replace("(", " ").replace(")", " ")
    return re.sub(r"[^A-Z0-9]+", " ", s).strip()


def _loose_name(s: str) -> str:
    return re.sub(r"\s+INDEX$", "", _norm_name(s)).strip()


def fetch_bse_dropdown(session: requests.Session) -> dict[str, str]:
    """Return {normalized-name: strIndex_code} for every BSE index in the dropdown.
    Registers both the tight and loose (no trailing INDEX) forms."""
    r = session.get(BSE_DROPDOWN_URL, headers=BSE_HEADERS, timeout=30)
    r.raise_for_status()
    out: dict[str, str] = {}
    for row in r.json()["Table"]:
        alias = row["shortalias"]
        code = row["Indx_cd"]
        out.setdefault(_norm_name(alias), code)
        out.setdefault(_loose_name(alias), code)
    return out


def resolve_bse_code(name: str, dropdown: dict[str, str]) -> str | None:
    return dropdown.get(_norm_name(name)) or dropdown.get(_loose_name(name))


def fetch_bse(session: requests.Session, code: str,
              start: datetime, end: datetime) -> pd.DataFrame | None:
    params = {
        "strIndex": code,
        "dtFromDate": start.strftime("%d/%m/%Y"),
        "dtToDate": end.strftime("%d/%m/%Y"),
    }
    try:
        r = session.get(BSE_CSV_URL, params=params, headers=BSE_HEADERS, timeout=60)
    except requests.RequestException as e:
        log.warning("BSE %s: HTTP error %s", code, e)
        return None
    if r.status_code != 200 or not r.text.strip() or "<html" in r.text[:200].lower():
        return None
    try:
        df = pd.read_csv(StringIO(r.text))
    except Exception:
        return None
    if df.empty:
        return None
    df.columns = [c.strip() for c in df.columns]
    date_col = next((c for c in df.columns if c.lower().startswith("date")), None)
    if not date_col:
        return None
    df = df.rename(columns={date_col: "Date"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["Date"]).sort_values("Date").set_index("Date")
    for c in ("Open", "High", "Low", "Close"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "", regex=False), errors="coerce")
    return add_return_and_vol(df)


# ---------------------------------------------------------------------------
# India VIX (via yfinance) — the niftyindices API caps VIX at 2018-01-01
# ---------------------------------------------------------------------------

def fetch_india_vix(start: datetime, end: datetime) -> pd.DataFrame | None:
    import yfinance as yf  # imported lazily so main path works if user hasn't installed it
    df = yf.download("^INDIAVIX", start=start.date().isoformat(),
                     end=(end.date().isoformat()), progress=False, auto_adjust=False)
    if df is None or df.empty:
        return None
    if hasattr(df.columns, "levels"):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df.dropna(subset=["Close"])
    df.index.name = "Date"
    keep = [c for c in ("Open", "High", "Low", "Close") if c in df.columns]
    return add_return_and_vol(df[keep].copy())


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def download_nse(session, indices, start, end, force, sleep) -> tuple[list, list]:
    session.get(NIFTY_WARMUP, headers={"User-Agent": USER_AGENT}, timeout=30)
    todo = [i for i in indices if i["exchange"] == "NSE"
            and (force or not os.path.exists(target_path(i["name"])))]
    log.info("NSE: %d to fetch", len(todo))
    ok, fail = [], []
    for i, idx in enumerate(todo, 1):
        nm = idx["name"]
        df = fetch_nifty(session, nm, start, end)
        if df is None or df.empty:
            log.warning("[NSE %d/%d] %s -- FAILED", i, len(todo), nm)
            fail.append(nm)
        else:
            df.to_csv(target_path(nm))
            log.info("[NSE %d/%d] %s -> %d rows", i, len(todo), nm, len(df))
            ok.append({"name": nm, "rows": len(df)})
        time.sleep(sleep)
    return ok, fail


def download_bse(session, indices, start, end, force, sleep) -> tuple[list, list, list]:
    dropdown = fetch_bse_dropdown(session)
    log.info("BSE dropdown: %d entries", len(set(dropdown.values())))
    todo = [i for i in indices if i["exchange"] == "BSE"
            and (force or not os.path.exists(target_path(i["name"])))]
    log.info("BSE: %d to fetch", len(todo))
    ok, fail, unmatched = [], [], []
    for i, idx in enumerate(todo, 1):
        nm = idx["name"]
        code = resolve_bse_code(nm, dropdown)
        if not code:
            log.info("[BSE %d/%d] %s -- no dropdown match", i, len(todo), nm)
            unmatched.append(nm)
            continue
        df = fetch_bse(session, code, start, end)
        if df is None or df.empty:
            log.warning("[BSE %d/%d] %s (%s) -- FAILED", i, len(todo), nm, code)
            fail.append({"name": nm, "code": code})
        else:
            df.to_csv(target_path(nm))
            log.info("[BSE %d/%d] %s (%s) -> %d rows", i, len(todo), nm, code, len(df))
            ok.append({"name": nm, "code": code, "rows": len(df)})
        time.sleep(sleep)
    return ok, fail, unmatched


def download_vix(start, end, force) -> bool:
    """Always fetch VIX from yfinance — the niftyindices VIX series is capped at
    2018-01-01, whereas yfinance ^INDIAVIX goes back to launch (2008-03).
    We overwrite whatever the NSE stage may have written."""
    path = os.path.join(DATA_DIR, "india_vix_yfinance.csv")
    df = fetch_india_vix(datetime(2008, 1, 1), end)
    if df is None or df.empty:
        log.warning("India VIX: failed")
        return False
    df.to_csv(path)
    log.info("India VIX -> %d rows %s..%s", len(df), df.index.min().date(), df.index.max().date())
    return True


# ---------------------------------------------------------------------------
# Report — data collection.md + status JSON
# ---------------------------------------------------------------------------

def _resolve_display_name(file_slug: str, csv_meta: list[tuple[str, str]]) -> tuple[str, str]:
    slug_map = {slugify(n): (n, e) for n, e in csv_meta}
    if file_slug in slug_map:
        return slug_map[file_slug]
    for n, e in csv_meta:
        if slugify(n).replace("-", "_") == file_slug.replace("-", "_"):
            return n, e
    if file_slug.startswith("nifty"):
        return file_slug.replace("_", " ").title(), "NSE"
    if file_slug.startswith("bse"):
        return file_slug.replace("_", " ").upper(), "BSE"
    return file_slug, "?"


def write_report(start: datetime, end: datetime) -> dict:
    csv_meta = [(x["name"], x["exchange"]) for x in read_indices()]
    csv_slugs = {slugify(n): (n, e) for n, e in csv_meta}
    have_slugs = {fn[:-len("_yfinance.csv")]
                  for fn in os.listdir(DATA_DIR) if fn.endswith("_yfinance.csv")}

    rows = []
    for slug in sorted(have_slugs):
        fn = f"{slug}_yfinance.csv"
        display, exch = _resolve_display_name(slug, csv_meta)
        try:
            df = pd.read_csv(os.path.join(DATA_DIR, fn), usecols=[0])
        except Exception as e:
            log.warning("Skip %s (%s)", fn, e); continue
        if df.empty:
            continue
        rows.append({
            "exchange": exch,
            "name": display,
            "start": str(df.iloc[0, 0])[:10],
            "end":   str(df.iloc[-1, 0])[:10],
            "rows":  len(df),
        })

    rows.sort(key=lambda r: (r["exchange"], r["name"].lower()))

    md = [
        "# Data Collection — Downloaded Indices\n",
        f"**Generated:** {datetime.now().date()}  ",
        f"**Files:** {len(rows)}  ",
        f"**Requested range:** {start.date()} → {end.date()}. Actual start = each index's own inception (base) date.\n",
        "| # | Exchange | Index | Start date | End date | Rows |",
        "|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(rows, 1):
        md.append(f"| {i} | {r['exchange']} | {r['name']} | {r['start']} | {r['end']} | {r['rows']:,} |")
    with open(os.path.join(BASE_DIR, "data collection.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    # Status JSON
    have_display = {slugify(r["name"]) for r in rows}
    nse_missing = [n for n, e in csv_meta if e == "NSE" and slugify(n) not in have_display]
    bse_missing = [n for n, e in csv_meta if e == "BSE" and slugify(n) not in have_display]
    status = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "total_in_csv": len(csv_meta),
        "downloaded": len(rows),
        "nse_missing": nse_missing,
        "bse_missing": bse_missing,
    }
    with open(os.path.join(DATA_DIR, "download_status.json"), "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)

    log.info("Report: %d/%d indices downloaded (%.1f%%)",
             len(rows), len(csv_meta), 100 * len(rows) / len(csv_meta))
    log.info("Missing: NSE=%d BSE=%d", len(nse_missing), len(bse_missing))
    return status


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--only", choices=("all", "nse", "bse", "vix", "report"),
                   default="all", help="Which stage to run (default: all)")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing CSVs instead of skipping them")
    p.add_argument("--start", type=parse_date, default=DEFAULT_START,
                   help="Start date YYYY-MM-DD (default 2000-01-01)")
    p.add_argument("--end", type=parse_date, default=DEFAULT_END,
                   help="End date YYYY-MM-DD (default today)")
    p.add_argument("--sleep", type=float, default=0.5,
                   help="Seconds between API calls (default 0.5)")
    args = p.parse_args(argv)

    session = build_session()

    if args.only in ("all", "nse"):
        indices = read_indices()
        ok, fail = download_nse(session, indices, args.start, args.end, args.force, args.sleep)
        log.info("NSE done: %d ok, %d failed", len(ok), len(fail))

    if args.only in ("all", "bse"):
        indices = read_indices()
        ok, fail, unmatched = download_bse(session, indices, args.start, args.end, args.force, args.sleep)
        log.info("BSE done: %d ok, %d failed, %d unmatched", len(ok), len(fail), len(unmatched))

    if args.only in ("all", "vix"):
        download_vix(args.start, args.end, args.force)

    write_report(args.start, args.end)
    return 0


if __name__ == "__main__":
    sys.exit(main())
