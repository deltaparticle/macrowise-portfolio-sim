# Indian Indices Data Download — Final Report

**Generated:** 2026-07-10
**Requested range:** 2000-01-01 → 2026-07-10 (per-index, actual first-date depends on when each index started)

## Coverage

| Segment | Downloaded | Total | % |
|---|---|---|---|
| NSE  | 144 (+ 1 alias) | 145 | **100%** effective |
| BSE  | 13  | 125 | 10% |
| **Total** | **157** | **270** | **58%** |

- `Nifty Healthcare Index` (the one nominal NSE miss) is just the formal name of `Nifty Healthcare`, which is already downloaded — so all 145 NSE indices are effectively covered.

## What changed this session
- NSE: went from 39 → 145 files. Root cause of prior 15% coverage was `nsepy` (deprecated) and `yfinance` (covers only ~40 Indian indices). Fixed by switching to `niftyindices.com`'s current endpoint (`/BackPage/getHistoricaldatatabletoString`) which returns full daily history in JSON.
- BSE: cleaned up **9 previously-corrupt files** (7 sectoral files that all held identical `^BSESN`/SENSEX data mislabeled as BANKEX/METAL/etc., plus 2 files I mis-mapped mid-session). Real data downloaded for 13 BSE indices with verified codes.

## BSE — still missing (112 of 125)

BSE's `ProduceCSVForDate/w` endpoint works and returns full history back to 2000, but requires an **exact short-code** per index (e.g. `SENSEX`, `BSE100`, `BSEMID`). BSE does not publish this mapping; it's baked into their Angular-SPA JS chunks and I couldn't extract it reliably through brute-force guessing without producing false matches (e.g. `BSE SENSEX 50` and `BSE SENSEX NEXT 50` both silently accepted `SENSEX` and returned duplicate data).

### To finish the BSE side, pick one of:

1. **Best (5-10 min manual):** Open <https://www.bseindia.com/indices/IndexArchiveData.html> in Chrome, open DevTools → Network, pick each missing index from the dropdown, download 1 day of data, and copy the `strIndex=` value from the request URL. Add pairs to `VERIFIED` in `bse_cleanup_and_refetch.py` and re-run.
2. **Scraper:** Fetch the archive-page's index dropdown via a headless browser (Playwright/Selenium) — the codes are populated at runtime by an Angular component. Not attempted here because it needs a JS engine.
3. **Skip:** Portfolio optimization work can proceed with the 157 downloaded indices — the missing BSE set is mostly sectoral duplicates of NSE sectorals you already have (e.g. `BSE AUTO` mirrors `Nifty Auto`, `BSE BANKEX` mirrors `Nifty Bank`).

See `data/download_status.json` for the full missing-list with the CSV's benchmark codes.

## Files

- `download_all_indices.py` — main downloader (NSE via niftyindices, BSE with verified code map)
- `bse_cleanup_and_refetch.py` — BSE cleanup + refetch script
- `download_status.py` — regenerates `data/download_status.json` and this report's numbers
- `data/*_yfinance.csv` — one file per index. Columns: `Date, Open, High, Low, Close, Return, Volatility`
- `data/download_status.json` — machine-readable coverage
