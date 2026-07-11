# Macrowise Portfolio Optimization Engine

A model-agnostic portfolio optimization engine for Indian indices (BSE + NSE).
The user states what they want (a return target, a risk cap, a drawdown limit,
a preferred sector, a view on a specific index — any combination). The engine
picks the right optimizer, runs it, checks feasibility, and returns an
optimized portfolio with a full diagnostic trail.

- **Universe**: 264 Indian indices (sector, size, style, and theme indices).
- **Models**: 10 optimizers — Mean-Variance (max Sharpe, min variance, target
  return, target volatility), CVaR, Max Drawdown, Risk Parity, Hierarchical
  Risk Parity, Black-Litterman, Max Sortino, Max Omega, Max Diversification,
  Inverse Volatility.
- **Estimation**: James-Stein shrunk mean + regime-aware covariance (Ledoit-Wolf in calm periods, EWMA under volatility stress)
  (industry standard).
- **Delivery**: Python module + JSON-driven CLI. No frontend dependency.

---

## Table of Contents

1. [What Problem This Solves](#1-what-problem-this-solves)
2. [Repository Layout](#2-repository-layout)
3. [Setup](#3-setup)
4. [Running the Engine](#4-running-the-engine)
5. [The Request Schema](#5-the-request-schema)
6. [12 Worked Examples with Full Results](#6-12-worked-examples-with-full-results)
7. [Models — Every Optimizer in Detail](#7-models--every-optimizer-in-detail)
8. [Model Selection Logic](#8-model-selection-logic)
9. [Estimators and Data Pipeline](#9-estimators-and-data-pipeline)
10. [Sector Taxonomy](#10-sector-taxonomy)
11. [Overall Test Results](#11-overall-test-results)
12. [Known Limitations and Data Realities](#12-known-limitations-and-data-realities)
13. [Future Improvements](#13-future-improvements)
14. [Architecture Notes](#14-architecture-notes)
15. [Reference Cards](#15-reference-cards)

---

## 1. What Problem This Solves

A user (retail or advisor) walks in with an intent expressed in one or more of
these forms:

- *"I want at least a 15% annual return."*
- *"I can tolerate 18% volatility at most."*
- *"Cap my max drawdown at 20%."*
- *"Limit tail loss (CVaR) to 2.5% per day."*
- *"Invest in IT and banking sectors only."*
- *"I have a view that IT will outperform Pharma by 4%."*
- *"Just give me a balanced portfolio."*
- Any combination of the above.

The engine translates this intent into a mathematical problem, picks the
optimizer that can solve it, runs it, and returns:

- Recommended weights per index.
- Realized-window metrics (return, vol, Sharpe, Sortino, drawdown, CVaR).
- Feasibility check against every user constraint.
- The full ranked candidate table so the user sees why this model was chosen.

If the user's request is infeasible, the engine returns the best-effort
portfolio and reports which constraints were violated by how much.

---

## 2. Repository Layout

```
D:/Macrowise Portfolio Optimization/
├── data/                            264 index CSVs from yfinance
├── data collection.md               data catalog: index + date range + rows
├── engine/                          the optimization engine
│   ├── __init__.py
│   ├── config.py                    constants (paths, TRADING_DAYS, RF, lookback)
│   ├── data_loader.py               CSV -> aligned price/return panel
│   ├── sector_map.py                regex tags for sector/size/style/theme
│   ├── estimators.py                mu (hist / JS / EWMA), cov (LW / sample / EWMA), CAPM-implied Pi
│   ├── metrics.py                   return, vol, Sharpe, Sortino, MaxDD, CVaR, Calmar
│   ├── selector.py                  UserRequest -> ordered list of candidate models -> ranked results
│   ├── optimizer.py                 orchestrator; the public entry point
│   ├── backtest.py                  walk-forward periodic-rebalance backtester
│   └── models/                      the 10 optimizers
│       ├── _common.py               shared: cvxpy constraints, solver retry, simplex projection
│       ├── mean_variance.py         min_variance, min_variance_for_return, max_return_for_vol, max_sharpe
│       ├── cvar.py                  min_cvar, min_cvar_for_return  (Rockafellar-Uryasev LP)
│       ├── drawdown.py              min_max_drawdown             (differential evolution)
│       ├── risk_parity.py           ERC via SLSQP
│       ├── hrp.py                   Hierarchical Risk Parity     (Lopez de Prado)
│       ├── black_litterman.py       Idzorek confidence + posterior; feeds max_sharpe / target_return
│       ├── sortino.py               max Sortino  (multi-start SLSQP)
│       ├── omega.py                 max Omega    (multi-start SLSQP)
│       ├── max_diversification.py   Choueifaty diversification ratio
│       └── inverse_vol.py           baseline
├── cli.py                           `optimize`, `backtest`, `sectors`, `indices` subcommands
├── examples/                        12 sample UserRequest JSON files
├── opt_goal_sector_test.py          the 360-case test runner
├── requirements.txt
└── README.md                        (this file)
```

---

## 3. Setup

### 3.1 Prerequisites

- Python 3.10 or newer
- `pip` (or `uv` / `pipx`)
- `git` (to clone the repository)
- ~200 MB free disk for the pre-downloaded index CSVs

### 3.2 Clone the repository

```bash
git clone https://github.com/deltaparticle/macrowise-portfolio-sim.git
cd macrowise-portfolio-sim
```

If you already have the repo and want the latest:

```bash
cd macrowise-portfolio-sim
git pull
```

### 3.3 Create a virtual environment (recommended)

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Windows (Git Bash)
python -m venv .venv
source .venv/Scripts/activate
```

### 3.4 Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` pins:

```
numpy>=1.24
pandas>=2.0
scipy>=1.11
scikit-learn>=1.3
cvxpy>=1.4
pyyaml>=6.0
tabulate>=0.9

# API
fastapi>=0.110
uvicorn[standard]>=0.27
pydantic>=2.5
```

`cvxpy` installs three convex solvers alongside it (ECOS, SCS, Clarabel). The
engine tries them in that order and uses whichever succeeds first.

### 3.5 Verify the install

```bash
python -c "from engine.optimizer import optimize, UserRequest; print('OK')"
python cli.py sectors
```

`python cli.py sectors` prints the full tag catalog. If you see JSON with 20
sectors, 7 sizes, 13 styles, 7 themes — installation is good.

---

## 4. Running the Engine

### 4.1 Command line

```bash
# 1. Optimize using a JSON request file
python cli.py optimize examples/request_max_sharpe.json

# 2. Optimize and save the full result to JSON
python cli.py optimize examples/request_max_sharpe.json --out result.json

# 3. Walk-forward backtest of the same request
python cli.py backtest examples/request_balanced.json --rebalance Q --lookback 3

# 4. List every available sector / size / style / theme tag
python cli.py sectors

# 5. List every index tagged with a given sector
python cli.py indices --sector IT
```

### 4.2 Python API

```python
from engine.optimizer import UserRequest, optimize

request = UserRequest(
    sectors=["IT", "Banks", "FMCG"],   # any tag from `python cli.py sectors`
    target_return=0.15,                # optional: minimum annualized return
    max_volatility=0.20,               # optional: annualized vol cap
    max_drawdown=0.25,                 # optional: max drawdown cap
    w_max=0.25,                        # optional: per-index cap (default 1.0)
    lookback_years=5,                  # data window
    primary_goal="max_sharpe",         # optional: nudge the selector
)

result = optimize(request)

print(result.chosen_model)     # e.g. "max_sharpe" or "min_max_drawdown"
print(result.feasible)         # bool
print(result.metrics)          # {"ann_return", "ann_vol", "sharpe", "sortino", "max_drawdown", "cvar", "calmar"}
print(result.weights)          # {"nifty_it": 0.18, "bse_bankex": 0.14, ...}
print(result.all_candidates)   # ranked list of all models tried
```

### 4.3 REST API (FastAPI)

The engine also ships with a FastAPI backend at `api/main.py` for frontend
integration. Endpoints:

| Method | Path | Purpose |
|---|---|---|
| GET  | `/`         | Service info + endpoint list |
| GET  | `/health`   | Health check (data-dir status, index count) |
| POST | `/optimize` | Run an optimization; body is an `OptimizeRequest` JSON |
| POST | `/backtest` | Walk-forward backtest; body wraps the same request plus `rebalance` + `lookback_years` |
| GET  | `/sectors`  | Full tag catalog |
| GET  | `/indices?sector=IT` | List indices in a tag |
| GET  | `/models`   | Metadata for every optimizer (family, solver, which constraints it supports) |
| GET  | `/examples` | Every example JSON bundled with the repo |
| GET  | `/docs`     | Swagger UI (auto-generated from Pydantic schemas) |
| GET  | `/redoc`    | ReDoc UI |
| GET  | `/openapi.json` | Machine-readable OpenAPI spec (for frontend type generation) |

#### Local development

```bash
# reload on file change
./run_api.sh
# or explicitly
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload --timeout-keep-alive 300
```

Open `http://localhost:8000/docs` to try requests interactively.

#### Production (uvicorn workers)

```bash
./run_api.sh prod
# or
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 2 --timeout-keep-alive 300
```

The 300-second keep-alive is important — the `min_max_drawdown` model uses
differential evolution and can take ~40 s on larger universes. Default
uvicorn timeouts will drop the connection.

#### Docker

```bash
docker build -t macrowise-portfolio-api .
docker run -p 8000:8000 macrowise-portfolio-api
```

Or with compose:

```bash
docker compose up -d
docker compose logs -f api
docker compose down
```

The image is `python:3.11-slim` + system deps for cvxpy solvers, ~600 MB.
The healthcheck hits `/health` every 30 s.

#### Example request

```bash
curl -X POST http://localhost:8000/optimize \
     -H 'Content-Type: application/json' \
     -d '{
       "sectors": ["IT", "Banks", "FMCG"],
       "target_return": 0.15,
       "max_volatility": 0.20,
       "w_max": 0.25,
       "lookback_years": 5
     }'
```

Returns:

```json
{
  "chosen_model": "max_return_for_vol",
  "feasible": true,
  "reason": "",
  "universe_size": 24,
  "weights": {"nifty_fmcg": 0.19, "bse_bankex": 0.14, "...": "..."},
  "metrics": {"ann_return": 0.18, "ann_vol": 0.19, "sharpe": 0.61,
              "sortino": 0.86, "max_drawdown": 0.16,
              "cvar": 0.024, "calmar": 1.13},
  "candidates": [{"model": "max_return_for_vol", "feasible": true, "...": "..."}],
  "universe": ["nifty_it", "bse_bankex", "..."]
}
```

Every field of `UserRequest` is optional; leave it out to accept the default.
Unknown fields are rejected with HTTP 422 (Pydantic `extra="forbid"`).

### 4.4 Backtest API

```python
from engine.backtest import walk_forward
from engine.optimizer import UserRequest

result = walk_forward(
    UserRequest(sectors=["Largecap", "Broad"], primary_goal="max_sharpe"),
    rebalance="Q",       # M / Q / A / W
    lookback_years=3,
    initial_capital=100.0,
)
print(result["metrics"])
result["equity_curve"].plot()   # if you have matplotlib
```

---

## 5. The Request Schema

Every field is optional. Leave fields out to accept the default.

| Field | Type | Default | What it does |
|---|---|---|---|
| `sectors` | `list[str]` | `None` | Tag filter. Any tag from `python cli.py sectors`. |
| `universe` | `list[str]` | `None` | Explicit slug list (overrides `sectors`). |
| `sector_match_all` | `bool` | `False` | If True, require an index to have ALL of the listed tags. Falls back to any-match if empty. |
| `primary_goal` | `str` | `None` | One of `max_sharpe`, `max_return`, `min_risk`, `balanced`, `min_tail_risk`, `min_drawdown`, `max_sortino`, `max_omega`, `max_diversification`, `inverse_vol`. |
| `target_return` | `float` | `None` | Minimum annualized return, e.g. 0.15. Becomes a hard constraint. |
| `max_volatility` | `float` | `None` | Annualized vol cap. |
| `max_drawdown` | `float` | `None` | Max drawdown cap (positive number). |
| `max_cvar` | `float` | `None` | Max daily 5%-CVaR (positive number). |
| `cvar_alpha` | `float` | `0.05` | Tail probability level for CVaR. |
| `w_min` | `float` | `0.0` | Per-index minimum weight. |
| `w_max` | `float` | `1.0` | Per-index maximum weight (soft cap in current build — see limitations). |
| `start`, `end` | `str` | `None` | Data window (`YYYY-MM-DD`). If unset, `lookback_years` back from latest. |
| `lookback_years` | `int` | `5` | Data window length. |
| `risk_free_rate` | `float` | `0.065` | Annualized RF for Sharpe / max_sharpe. |
| `views` | `list[dict]` | `[]` | Black-Litterman views. Two shapes: `{"asset": "nifty_it", "return": 0.20, "confidence": 0.65}` or `{"long": ["nifty_bank"], "short": ["nifty_pharma"], "return": 0.04, "confidence": 0.5}`. Presence forces the Black-Litterman solver. |
| `market_weights` | `pd.Series` | `None` | Optional BL prior weights. Falls back to a broad-market index if present in universe, else inverse-vol proxy. |
| `bl_tau` | `float` | `0.05` | BL prior uncertainty scalar. |
| `top_k` | `int` | `3` | How many candidate models to keep in the output. |

---

## 6. 12 Worked Examples with Full Results

Every example below can be reproduced with:

```bash
python cli.py optimize examples/<file>.json
```

All results are computed on a 5-year lookback ending 2026-07-09. Numbers come
straight from `example_results.json`. **Negative Sharpe values in several
examples are data-real** — the 2021–2026 window includes a broad Indian
equity correction (2022) and a soft IT / broad-market patch (2023–24). This
is not a model bug.

---

### Example 01 — Max Sharpe on a five-sector mix

**File:** `examples/request_max_sharpe.json`

```json
{
  "sectors": ["IT", "Banks", "Auto", "Pharma", "FMCG"],
  "primary_goal": "max_sharpe",
  "w_max": 0.30, "lookback_years": 5,
  "risk_free_rate": 0.065
}
```

| Metric | Value |
|---|---|
| Chosen model | `risk_parity` |
| Feasible | Yes |
| Universe | 24 indices |
| Holdings | 24 |
| Ann. return | 0.97 % |
| Ann. vol | 14.12 % |
| Sharpe | −0.39 |
| Max drawdown | 15.36 % |

**Top holdings:** `nifty_fmcg` 6.59 %, `bse_fast_moving_consumer_goods` 6.20 %,
`nifty_pharma` 5.25 %, `nifty500_healthcare` 5.00 %, `bse_healthcare` 4.86 %.

**Candidate ranking:**

| Model | Feasible | Ret | Vol | Sharpe | MaxDD |
|---|---|---|---|---|---|
| risk_parity  | Yes |  0.97 % | 14.12 % | −0.39 | 15.36 % |
| min_variance | Yes | −2.09 % | 11.71 % | −0.73 | 14.24 % |

**Reading it:** the selector's fallback pool for `max_sharpe` includes
`risk_parity`. Because JS-shrunk means over this sample are compressed,
`risk_parity` produced a higher-Sharpe portfolio than raw `max_sharpe`, so
the engine chose it.

---

### Example 02 — Target return 15 % with 22 % vol cap

**File:** `examples/request_target_return_vol_cap.json`

```json
{
  "sectors": ["IT", "Banks", "FinancialServices", "Pharma", "FMCG", "Auto"],
  "target_return": 0.18, "max_volatility": 0.22,
  "w_max": 0.25, "lookback_years": 5
}
```

| Metric | Value |
|---|---|
| Chosen model | `max_return_for_vol` |
| Feasible | No *(return 5.19 % < target 18 %)* |
| Universe | 35 |
| Ann. return | 5.19 % |
| Ann. vol | 15.61 % |

**Verdict:** engine correctly reports infeasibility with the exact violated
constraint. The best-effort portfolio still delivers 5.19 % / 15.6 % vol —
useful for a user to see how far off their target is.

---

### Example 03 — Cap drawdown at 20 %, floor return at 12 %

**File:** `examples/request_max_dd.json`

```json
{
  "sectors": ["Largecap", "Broad", "FMCG", "Pharma"],
  "primary_goal": "min_drawdown",
  "max_drawdown": 0.20, "target_return": 0.12,
  "lookback_years": 5
}
```

| Metric | Value |
|---|---|
| Chosen model | `min_max_drawdown` (differential evolution) |
| Feasible | Yes |
| Universe | 50 |
| Holdings | 8 |
| Ann. return | 14.80 % |
| Ann. vol | 12.64 % |
| Sharpe | 0.66 |
| Sortino | 0.92 |
| Max drawdown | **8.00 %** |
| CVaR (5 %) | 1.82 % |

**Top holdings:** `nifty_pharma` 21.74 %, `nifty500_healthcare` 21.15 %,
`bse_healthcare` 21.04 %, `nifty_midsmall_healthcare` 20.20 %,
`bse_500_quality_50` 6.71 %, `bse_500_dividend_leaders_50` 6.10 %.

**Reading it:** the non-convex DE solver finds a healthcare-heavy tilt that
delivers a well-above-floor 14.8 % return with only 8 % max drawdown — well
inside the 20 % cap.

---

### Example 04 — Black-Litterman with two views

**File:** `examples/request_black_litterman.json`

```json
{
  "sectors": ["IT", "Banks", "Auto", "Pharma", "FMCG", "Energy"],
  "views": [
    {"asset": "nifty_it", "return": 0.20, "confidence": 0.65},
    {"long": ["nifty_bank"], "short": ["nifty_pharma"], "return": 0.04, "confidence": 0.5}
  ],
  "bl_tau": 0.05, "w_max": 0.30, "lookback_years": 5
}
```

| Metric | Value |
|---|---|
| Chosen model | `black_litterman` |
| Feasible | Yes (no numeric target set) |
| Universe | 32 |
| Holdings | 6 |
| Ann. return | −15.07 % |
| Ann. vol | 20.20 % |
| Sharpe | −1.07 |
| Max drawdown | 28.94 % |

**Top holdings:** `nifty_it` 30 %, `bse_focused_it` 30 %, `nifty_india_internet`
19.36 %, `nifty_financial_services_ex_bank` 9.23 %, `bse_information_technology`
7.65 %, `bse_psu_bank` 3.76 %.

**Reading it:** the model faithfully implements the bullish-IT view. The
negative realized return is a *view-quality* problem, not a model problem: the
posterior went heavy on IT during a 5-year window where Indian IT indices
underperformed. This is the expected BL behaviour — the user should override
`market_weights`, tune confidences, or shorten `bl_tau` if the prior is
dominating.

---

### Example 05 — Balanced portfolio (Risk Parity / HRP)

**File:** `examples/request_balanced.json`

```json
{
  "sectors": ["Largecap", "Midcap", "Broad"],
  "primary_goal": "balanced",
  "lookback_years": 5
}
```

| Metric | Value |
|---|---|
| Chosen model | `risk_parity` |
| Feasible | Yes |
| Universe | 52 |
| Holdings | 52 (fully diversified) |
| Ann. return | 2.41 % |
| Ann. vol | 13.61 % |
| Max drawdown | 14.34 % |

**Top holdings:** all in the 2.0 – 2.7 % range — `nifty50_shariah`,
`nifty50_value_20`, `nifty100_low_volatility_30`, `nifty100_quality_30`,
`bse_500_low_volatility_50`, `bse_sensex`, `nifty_50`.

**Candidate table:**

| Model | Ret | Vol | Sharpe |
|---|---|---|---|
| risk_parity           |  2.41 % | 13.61 % | −0.30 |
| hrp                   |  1.70 % | 13.38 % | −0.36 |
| max_diversification   | −0.27 % | 13.27 % | −0.51 |

**Reading it:** `balanced` fans out into RP, HRP, and MaxDiv. RP wins because
its scoring (Sharpe by default) is highest. This is a genuinely diversified
allocation — no single index above 2.7 %.

---

### Example 06 — Minimize tail risk (CVaR ≤ 2.5 %)

**File:** `examples/request_tail_risk.json`

```json
{
  "sectors": ["Broad", "Largecap", "FMCG", "Pharma"],
  "primary_goal": "min_tail_risk",
  "max_cvar": 0.025, "cvar_alpha": 0.05,
  "target_return": 0.14, "lookback_years": 5
}
```

| Metric | Value |
|---|---|
| Chosen model | `min_cvar` |
| Feasible | No *(return 1.31 % < target 14 %)* |
| Universe | 50 |
| Holdings | 4 |
| Ann. return | 1.31 % |
| Ann. vol | 10.61 % |
| Max drawdown | 11.99 % |
| CVaR (5 %) | **1.55 %** (well under 2.5 % cap) |

**Top holdings:** `nifty500_healthcare` 40.51 %, `nifty50_value_20` 33.57 %,
`nifty_fmcg` 22.58 %, `nifty_hospitals` 3.34 %.

**Reading it:** CVaR minimization achieves a very low tail loss but the
target return of 14 % is not reachable within this universe over this window.
The engine flags the exact violation.

---

### Example 07 — Min risk on a broad Largecap universe

**File:** `examples/ex07_min_risk_broad.json`

```json
{
  "sectors": ["Largecap", "Broad"],
  "primary_goal": "min_risk",
  "w_max": 0.25, "lookback_years": 5
}
```

| Metric | Value |
|---|---|
| Chosen model | `risk_parity` |
| Feasible | Yes |
| Universe | 43 |
| Holdings | 43 |
| Ann. return | 1.17 % |
| Ann. vol | 13.24 % |
| Max drawdown | 14.27 % |

**Reading it:** raw `min_variance` in the candidate table returned −4.29 %
at 11.34 % vol. The **return-floor penalty** in the selector's `min_risk`
scoring correctly penalized that solution and promoted `risk_parity`, which
delivers positive expected return at only slightly higher vol.

---

### Example 08 — Max Sortino on a six-sector universe

**File:** `examples/ex08_max_sortino.json`

```json
{
  "sectors": ["IT", "Banks", "FinancialServices", "Auto", "Pharma", "FMCG"],
  "primary_goal": "max_sortino",
  "w_max": 0.25, "lookback_years": 5
}
```

| Metric | Value |
|---|---|
| Chosen model | `max_sortino` |
| Feasible | Yes |
| Universe | 35 |
| Holdings | 5 |
| Ann. return | **24.50 %** |
| Ann. vol | 20.18 % |
| Sharpe | 0.89 |
| Sortino | **1.29** |
| Max drawdown | 14.33 % |

**Top holdings:** `nifty_midsmall_financial_services` 25 %, `nifty_nbfc` 25 %,
`nifty_hospitals` 25 %, `nifty_capital_markets` 19.22 %, `nifty_psu_bank`
5.78 %.

**Reading it:** downside-only risk penalization concentrates into financials
and healthcare, delivering the highest return of the 12 examples. Concentrated
(5 names) but each is a diversified sub-index, so intra-index diversification
still exists.

---

### Example 09 — Max Diversification across size buckets

**File:** `examples/ex09_max_diversification.json`

```json
{
  "sectors": ["Largecap", "Midcap", "Smallcap", "Broad"],
  "primary_goal": "max_diversification",
  "w_max": 0.15, "lookback_years": 5
}
```

| Metric | Value |
|---|---|
| Chosen model | `risk_parity` |
| Feasible | Yes |
| Universe | 59 |
| Holdings | 59 (fully diversified) |
| Ann. return | 3.13 % |
| Ann. vol | 13.82 % |
| Max drawdown | 14.42 % |

**Candidate table:**

| Model | Ret | Vol | Sharpe |
|---|---|---|---|
| risk_parity          | 3.13 % | 13.82 % | −0.24 |
| max_diversification  | 2.29 % | 13.49 % | −0.31 |

**Reading it:** Choueifaty's DR objective and RP produce similar allocations
here — the correlation matrix is quite uniform across Indian size buckets,
so DR has less room to add value than in a mixed-asset universe.

---

### Example 10 — Combined constraints (return + vol + DD)

**File:** `examples/ex10_multi_constraint.json`

```json
{
  "sectors": ["Largecap", "Midcap", "Broad", "FMCG", "Pharma"],
  "target_return": 0.15, "max_volatility": 0.18, "max_drawdown": 0.25,
  "w_max": 0.20, "lookback_years": 5
}
```

| Metric | Value |
|---|---|
| Chosen model | `min_max_drawdown` |
| Feasible | No *(return 14.70 % < target 15 %, missed by 30 bps)* |
| Universe | 59 |
| Holdings | 9 |
| Ann. return | 14.70 % |
| Ann. vol | 12.50 % |
| Sharpe | 0.66 |
| Max drawdown | 8.06 % |

**Reading it:** the drawdown-solver was pulled into service because
`max_drawdown` was set. Two of three constraints (vol 12.5 %, DD 8.1 %) are
comfortably inside caps; the return misses by 0.3 %. The engine reports
"almost feasible" rather than silently returning an infeasible portfolio.

---

### Example 11 — Factor-index universe

**File:** `examples/ex11_factor_indices.json`

```json
{
  "sectors": ["Momentum", "Quality", "LowVolatility", "Value"],
  "primary_goal": "max_sharpe",
  "w_max": 0.20, "lookback_years": 5
}
```

| Metric | Value |
|---|---|
| Chosen model | `risk_parity` |
| Feasible | Yes |
| Universe | 31 factor indices |
| Ann. return | 1.36 % |
| Ann. vol | 13.25 % |
| Max drawdown | 13.16 % |

**Top holdings:** dominated by low-volatility variants — `nifty50_value_20`,
`nifty_quality_low_volatility_30`, `bse_low_volatility_index`,
`nifty100_low_volatility_30`.

**Reading it:** the sector-map correctly identifies factor indices (Momentum,
Quality, LowVol, Value) through name matching, letting you build factor-tilted
portfolios directly.

---

### Example 12 — Degenerate universe (single sector, single usable index)

**File:** `examples/ex12_defence_degenerate.json`

```json
{
  "sectors": ["Defence"],
  "primary_goal": "max_sharpe",
  "lookback_years": 5
}
```

| Metric | Value |
|---|---|
| Chosen model | `single_asset_trivial` |
| Feasible | Yes |
| Universe | 1 (`nifty_india_defence` only — `bse_india_defence` filtered for short history) |
| Ann. return | 60.64 % |
| Ann. vol | 27.48 % |
| Sharpe | 1.97 |
| Max drawdown | 38.21 % |

**Reading it:** the Defence sector has only one index with enough history to
enter the optimizer. Instead of a solver crash, the engine returns
`w = [1.0]` and reports the reason. This was a bug fixed in the second-round
cleanup (KeyError on `assets` when all candidate solvers failed on a
single-asset problem).

---

## 7. Models — Every Optimizer in Detail

The engine ships with **10 distinct optimizers**, plus one degenerate-case
handler. Each has a defined interface and is selected automatically for the
constraint mix in the request.

### 7.1 Mean-Variance family (`engine/models/mean_variance.py`)

The Markowitz workhorse. Four solvable forms exposed:

| Function | Objective | Constraints |
|---|---|---|
| `min_variance(mu, cov)` | `min w' Σ w` | long-only, weight bounds |
| `min_variance_for_return(mu, cov, target)` | `min w' Σ w` | + `μ' w ≥ target` |
| `max_return_for_vol(mu, cov, max_vol)` | `max μ' w` | + `w' Σ w ≤ max_vol²` |
| `max_sharpe(mu, cov, rf)` | `max (μ' w − rf) / √(w' Σ w)` | convex reformulation via variable substitution |

All four are solved as **convex QPs / SOCPs** using CVXPY. The solver stack
tries `ECOS → SCS → Clarabel` and falls back if one errors.

Max-Sharpe is a classic non-convex ratio. Solved here by the standard
reformulation:

```
min   y' Σ y
s.t.  (μ − rf 1)' y = 1
      Σ y = κ
      w_min · κ ≤ y ≤ w_max · κ
w = y / κ                          (recovered after solve)
```

If `all(μ ≤ rf)` (no risky asset beats the risk-free rate) it degenerates to
`min_variance`.

### 7.2 Conditional Value-at-Risk (`engine/models/cvar.py`)

Rockafellar & Uryasev's LP formulation:

```
min   ζ + (1 / (T α)) Σ u_t
s.t.  u_t ≥ -R_t w - ζ,   u_t ≥ 0
      long-only + weight bounds
      optional:  μ' w ≥ target_return
```

Solved as an LP. `α` defaults to 0.05 (5 % tail).

### 7.3 Max Drawdown (`engine/models/drawdown.py`)

Portfolio max drawdown is a non-smooth path functional of the equity curve.
No closed-form convex reduction — solved with **SciPy's differential
evolution** over the simplex:

```
min   MaxDD(cumprod(1 + R w))
      + λ · max(0, target_return − μ' w)    (soft penalty if target set)
```

Weights are projected onto the bounded simplex after each candidate. Slow
(seconds to a minute for larger universes); used only when the user actually
sets `max_drawdown` or picks `min_drawdown`.

### 7.4 Risk Parity — Equal Risk Contribution (`engine/models/risk_parity.py`)

Spinu / Bruder-Roncalli formulation, solved with SLSQP:

```
min   Σ_i ( w_i · (Σ w)_i / √(w' Σ w) − √(w' Σ w) / N )²
s.t.  Σ w = 1,  w in [w_min, w_max]
```

Does not require expected returns — pure covariance-based. Well-behaved,
diversified allocations.

### 7.5 Hierarchical Risk Parity (`engine/models/hrp.py`)

López de Prado's HRP. Three steps:

1. Compute correlation-distance matrix.
2. Single-linkage clustering + quasi-diagonalization.
3. Recursive bisection with inverse-variance weighting inside each split.

No inverse-covariance needed → stable on ill-conditioned covariance matrices.

### 7.6 Black-Litterman (`engine/models/black_litterman.py`)

Full Idzorek-style implementation:

1. **Prior**: reverse-optimize implied equilibrium returns
   `Π = δ · Σ · w_mkt` where `δ` is derived from `market_excess_return` and
   `market_variance`.
2. **Views**: user's list is compiled into `P` and `Q`. Two shapes supported:
   - Absolute: `{asset, return, confidence}` → row of `P` with a single 1.
   - Relative: `{long, short, return, confidence}` → `P` row sums to 0.
3. **View uncertainty Ω**: Idzorek-lite mapping
   `Ω_ii = (1/c_i − 1) · τ · p_i' Σ p_i`.
4. **Posterior**:
   ```
   μ_BL = (⁧(τΣ)⁻¹ + P' Ω⁻¹ P⁨)⁻¹ · (⁧(τΣ)⁻¹ Π + P' Ω⁻¹ Q⁨)
   Σ_BL = Σ + (⁧(τΣ)⁻¹ + P' Ω⁻¹ P⁨)⁻¹
   ```
5. **Optimize**: feed `(μ_BL, Σ_BL)` to `max_sharpe` (or `min_variance_for_return`
   if `target_return` set).

Default `market_weights` fallback (when no broad-market index is in the
universe): **inverse-vol proxy** (approximates cap-weighting) instead of 1/N.

### 7.7 Max Sortino (`engine/models/sortino.py`)

```
max   √252 · mean(R w − MAR) / √mean(min(R w − MAR, 0)²)
```

Non-convex — SLSQP with **multi-start** (3 starts: uniform + 2 Dirichlet
draws with fixed seeds) to escape local optima. MAR derived from
`target_return` if set, else 0.

### 7.8 Max Omega (`engine/models/omega.py`)

```
max   E[max(R w − MAR, 0)] / E[max(MAR − R w, 0)]
```

Same solver approach as Sortino.

### 7.9 Max Diversification (`engine/models/max_diversification.py`)

Choueifaty diversification ratio:

```
max   (w' σ) / √(w' Σ w)
```

where `σ` is the vector of per-asset standard deviations. SLSQP-solved.

### 7.10 Inverse Volatility (`engine/models/inverse_vol.py`)

Baseline: `w_i ∝ 1/σ_i`. Cheap, deterministic, no optimizer call. Useful as
a candidate against which the sophisticated models are ranked.

### 7.11 `single_asset_trivial`

Not a real optimizer — a guardrail. Kicks in when the universe collapses to
one index after history filtering (e.g., the Defence sector). Returns
`w = [1.0]` with a warning reason string.

---

## 8. Model Selection Logic

Implemented in `engine/selector.py`. The selector converts a `UserRequest`
into an **ordered list of candidate models**, runs each, checks feasibility,
scores them, and returns the ranked list.

### 8.1 Which model gets tried

Priority rules in order:

1. **Views present** → `black_litterman` only (views are always honored).
2. **Explicit `primary_goal`** → that model, plus a fallback pool:

   | Goal | Candidate pool |
   |---|---|
   | `max_sharpe` | max_sharpe, risk_parity |
   | `max_return` | max_sharpe (unbounded true "max return" is degenerate) |
   | `min_risk` | min_variance, risk_parity, max_sharpe |
   | `balanced` | risk_parity, hrp, max_diversification |
   | `min_tail_risk` | min_cvar, min_variance, max_sharpe |
   | `min_drawdown` | min_max_drawdown, risk_parity, max_sharpe |
   | `max_sortino` | max_sortino, max_sharpe |
   | `max_omega` | max_omega, max_sharpe |
   | `max_diversification` | max_diversification, risk_parity |
   | `inverse_vol` | inverse_vol, risk_parity |

3. **No goal, only constraints** → inferred:

   | Constraints present | Candidates |
   |---|---|
   | `max_drawdown` | min_max_drawdown, risk_parity, min_variance |
   | `max_cvar` + `target_return` | min_cvar_for_return, min_cvar, min_variance_for_return |
   | `max_cvar` only | min_cvar, min_variance |
   | `target_return` + `max_volatility` | max_return_for_vol, min_variance_for_return |
   | `target_return` only | min_variance_for_return, max_sharpe |
   | `max_volatility` only | max_return_for_vol, min_variance |
   | *(nothing)* | max_sharpe, risk_parity, hrp |

### 8.2 Scoring (`_score` in `selector.py`)

- If infeasible → large negative score plus per-violation penalty.
- If feasible:
  - `max_return` → `ann_return`
  - `min_risk` → `−ann_vol − 0.5 · max(0, rf − ann_return)`  (return-floor penalty)
  - `min_tail_risk` → `−cvar − 0.05 · max(0, rf − ann_return)`
  - `min_drawdown` → `−max_drawdown − 0.5 · max(0, rf − ann_return)`
  - `max_sortino` → `sortino`
  - `max_omega` → `calmar` (used as omega proxy; direct omega would be preferable long-term)
  - anything else → `sharpe`

The return-floor penalty on risk-only goals prevents a solver that hits a
low-vol / low-CVaR / low-DD outcome by loading up negative-return assets
from beating a slightly-higher-risk but positive-return alternative.

---

## 9. Estimators and Data Pipeline

### 9.1 Data pipeline (`engine/data_loader.py`)

1. Read the requested indices' `*_yfinance.csv` files.
2. Normalize timezones (some CSVs are tz-aware, some naive).
3. Filter to `[start, end]`.
4. Drop indices with fewer than `MIN_HISTORY_DAYS` (252) points.
5. Restrict to date range where ≥90 % of surviving indices have data.
6. Forward-fill up to 5 days, drop any residual NaNs.
7. Convert to simple returns (or log returns if `log_returns=True`).

### 9.2 Estimators (`engine/estimators.py`)

**Expected returns (μ):**

- `historical_mean` — simple sample mean, annualized.
- `ewma_mean(halflife=63)` — EWMA-weighted; recent data emphasized.
- `james_stein_mean` — **default in the optimizer.** Positive-part
  James-Stein shrinkage of the sample mean toward the grand mean:
  ```
  shrink   = min(1, max(0, (N−2) · mean(σ² / T) / Σ(μ̂ − μ̄)²))
  μ_shrunk = shrink · μ̄ + (1 − shrink) · μ̂
  ```
  Dominates the sample mean in MSE for N ≥ 3 (James-Stein 1961). The
  standard defense against mean-variance concentration driven by
  sample-mean noise.

**Covariance (Σ):**

- `sample_cov` — plain sample covariance.
- `ledoit_wolf_cov` — Shrinkage estimator from
  `sklearn.covariance.LedoitWolf`. Blends the sample covariance with a
  scaled identity, minimizing Frobenius-norm distance to the true
  covariance. Standard for portfolio optimization since Ledoit & Wolf (2003).
- `ewma_cov(halflife=63)` — exponentially weighted. Weights recent
  observations more heavily; tracks the current volatility regime rather
  than averaging over the whole lookback.
- `adaptive_cov` — **default in the optimizer (`cov_method="auto"`).**
  Runs `vol_regime_ratio()` (median of recent 30-day vol / full-window vol
  across assets) and switches to EWMA when the ratio exceeds
  `regime_threshold` (default 1.3), otherwise uses Ledoit-Wolf. Prevents
  Ledoit-Wolf from averaging stale calm-period data into risk estimates
  during a volatility spike.

Users can force a specific estimator via
`UserRequest(cov_method="ledoit_wolf" | "ewma" | "auto")`. The result
object reports `cov_method_used` and `vol_regime_ratio` so the caller can
see which mode fired.

**Walk-forward validation (Q rebalance, 3yr lookback, 2019–2024):**

| Mode | AnnRet | AnnVol | Sharpe | MaxDD |
|---|---:|---:|---:|---:|
| ledoit_wolf | 11.71% | 13.47% | 0.387 | 18.35% |
| ewma (always) | 10.92% | 13.61% | 0.325 | 19.22% |
| **auto** | **11.76%** | **13.46%** | **0.391** | **18.35%** |

Auto slightly dominates LW-only; always-EWMA underperforms both because
noise cost dominates in calm regimes. Auto fired 3 of 24 quarters (COVID
crash, 2019 mid-year, 2024 mid-year vol events).

**CAPM-implied equilibrium returns:**

- `implied_equilibrium_returns(cov, w_mkt)` — computes
  `Π = δ · Σ · w_mkt` with `δ` derived from a target
  `market_excess_return / market_variance`. Feeds the Black-Litterman prior.

---

## 10. Sector Taxonomy

`engine/sector_map.py` classifies every index by regex against its display
name, tagging four independent dimensions:

- **Sector** (20): IT, Banks, Auto, Pharma, FMCG, FinancialServices,
  Consumer, Energy, Metal, Realty, Infrastructure, Telecom, Media, Defence,
  Manufacturing, Chemicals, Commodities, Services, Rural, MNC.
- **Size** (7): Largecap, Midcap, Smallcap, Microcap, MidSmallcap, Multicap,
  Broad.
- **Style** (13): Momentum, Quality, Value, LowVolatility, Alpha, Dividend,
  Growth, HighBeta, MultiFactor, EqualWeight, Liquid, IPO, Focused.
- **Theme** (7): ESG, Shariah, PSU, CorporateGroup, Digital, Housing,
  Sectoral.

Additionally, indices matching `inverse`, `leverage`, `futures`, `vix`,
`usd`, `dividend points`, `2x`, `1x inverse` are marked `exclude=True` and
dropped from the default optimization pool (leveraged/inverse products are
usually not appropriate for buy-and-hold optimization).

Look up all tags:

```bash
python cli.py sectors
python cli.py indices --sector IT
```

Multi-tag filter: pass a list. Behaviour:

- `sector_match_all=False` (default) → any tag matches.
- `sector_match_all=True` → all tags must be on the same index; if empty,
  falls back to any-tag match to avoid empty universes.

---

## 11. Overall Test Results

A 360-case regression suite (`opt_goal_sector_test.py`) exercises every
combination of:

- 10 primary goals (`max_sharpe`, `min_risk`, `max_return`, `balanced`,
  `min_tail_risk`, `min_drawdown`, `max_sortino`, `black_litterman`,
  `inverse_vol`, `max_diversification`)
- 9 sector sets (single sector, sector pairs, cyclical mix, broad mix, etc.)
- 2 sector-match modes (any / all)
- 2 weight caps (0.30, 0.20)

**Second-round test outcome after the fixes described below:**

| Flag | First run | After fixes |
|---|---|---|
| `No indices tagged` (exception) | 120 | **0** |
| `weight_cap_breach` (FP tolerance) | 116 | 16 |
| `very_concentrated` (n ≤ 2 holdings) | 54 | 38 |
| `very_high_sharpe` (> 1.5) | 80 | 60 |
| `very_high_return` (> 50 %) | 68 | 56 |
| `errors ('assets' KeyError)` | 0 → 4 (regression) | **0** |
| `infeasible` | 18 → 38 (data-real, defence-heavy) | 38 |

**Fixes applied during the review round** (Section 12 covers each in more
detail):

1. **James-Stein mean shrinkage** (`estimators.py`) — used as default μ.
   Reduced max-Sharpe concentration from 4 positions at 30 % each to 20+
   positions at 3–5 %.
2. **Improved BL market_weights fallback** (`optimizer.py`) — inverse-vol
   proxy when no broad-market index is in the universe.
3. **Return-floor penalty in scoring** (`selector.py`) — prevents min-risk
   solvers from winning with negative-return solutions.
4. **Wider fallback pools** for risk-only goals (`selector.py`) — include
   `max_sharpe` so risk-adjusted alternatives can compete.
5. **Degenerate universe handler** (`optimizer.py`) — returns `single_asset_trivial`
   with `w = [1.0]` when history filtering leaves one index.
6. **Safer candidate selection** (`optimizer.py`) — filter error entries
   before picking `top[0]`; use `.get("assets", [])`.

**What the remaining flags actually mean:**

- `weight_cap_breach` — 3rd-decimal floating-point excess (0.300003 > 0.300000).
  Deliberately not fixed in this build per instruction.
- `very_high_return`, `very_high_vol`, `very_high_sharpe` — data-real. Banks
  and Defence rallied hard over the 5-year window; the metrics reflect
  reality.
- `very_concentrated` — single-sector universes with only 1–3 usable indices
  are inherently concentrated. Not a model bug.
- `infeasible` — small, degenerate universes (Defence, single sector) cannot
  simultaneously satisfy return + vol + DD constraints.

---

## 12. Known Limitations and Data Realities

- **Weight cap floating-point tolerance.** `w_max = 0.30` may see a weight
  come back as `0.300003` due to solver precision. Not corrected in this
  build.
- **Non-convex solvers use SciPy defaults.** Max Drawdown (DE), Sortino,
  Omega, Max-Div, and Risk Parity rely on iterative solvers with bounded
  iterations. Occasional suboptimal solutions on very large universes.
- **CVaR α is hard-coded to whatever the user sets (default 0.05).** No
  path yet to blend CVaR across multiple alphas.
- **Group / sector-cap constraints are not enforced in the solver yet.**
  You can filter the universe by sector, but you cannot say
  "no more than 40 % in Banks". Model-side changes are in the roadmap.
- **Turnover / transaction cost is not modelled.** Backtest currently
  rebalances free.
- **Historical mean is a poor point estimate.** JS shrinkage helps but does
  not fix the fundamental issue that 5-year sample means over Indian
  equities in 2021–2026 are noisy.
- **Some indices have very short history.** BSE India Defence (2025+), many
  BSE 500 factor variants (2025+). These get filtered out of most universes
  under the 252-day minimum, so multi-sector requests may return smaller
  universes than expected.
- **BL views require thoughtful confidence calibration.** A high-confidence
  view against a strong contrary prior will dominate. The engine will follow
  the math without warning about view quality.

---

## 13. Future Improvements

Prioritized:

1. **Group / sector-cap constraints inside solvers.**
   Add `group_limits: {"Banks": 0.40, "IT": 0.30}` to `UserRequest`; wire
   into every convex model as extra linear constraints, and into non-convex
   models as penalty terms.
2. **Michaud resampling.** Run mean-variance N times on bootstrap-resampled
   inputs, average the weights. The standard cure for MV point-estimate
   fragility — Portfolio Visualizer uses it as a checkbox.
3. **Turnover / transaction cost.** Add `w_prev` to `UserRequest` and a
   `turnover_penalty` field; augment the objective with `+ λ · ||w − w_prev||₁`.
4. **Cardinality constraints.** Cap the number of holdings (e.g., "give me
   at most 10 positions"). Requires mixed-integer solvers or a
   branch-and-cut approach.
5. **Natural-language request parser.** Thin LLM layer that converts free
   text ("give me a 15 % return portfolio with limited banking exposure")
   into a `UserRequest` dict. Keeps the core engine language-model-free.
6. **Factor-exposure optimization.** Match target factor loadings
   (Fama-French, Carhart, or custom regressions). Portfolio Visualizer's
   factor-based optimizer as a design reference.
7. **Direct market-cap ingestion for BL prior.** Currently defaults to
   inverse-vol when no broad-market index is present; a proper cap-weight
   feed (e.g., a monthly snapshot) would tighten the equilibrium prior.
8. **Efficient-frontier endpoint.** Sample the frontier at N target-return
   levels, return the (vol, return, weights) tuple for each — for UI plotting.
9. **Rebalance-aware backtest.** Include transaction costs, taxes, and a
   drift-aware rebalancing rule (e.g., only rebalance if any weight has
   drifted by more than X %).
10. **Stress-test scenarios.** Re-price the portfolio under historical crisis
    windows (2008, 2013 taper, 2020 COVID, 2022 correction) and report worst
    drawdown / recovery time.

---

## 14. Architecture Notes

**Design principles:**

- **One request → one result → many candidates.** The user shouldn't need
  to guess which optimizer to run. The engine tries several and ranks them
  transparently.
- **Feasibility over silence.** Every constraint violation is reported
  quantitatively (`return 0.1470 < target 0.1500` instead of "infeasible").
- **Composable inputs.** Any constraint can be combined with any other;
  the selector handles the mix.
- **Deterministic sector taxonomy.** Regex-based tagging is auditable;
  users can override with `universe=[...]` if they don't like the tag list.
- **Convex where possible, non-convex where necessary.** MV, CVaR, and BL
  are convex. Drawdown, Sortino, Omega, MaxDiv, RP are not — solved with
  SLSQP / DE and multi-start.
- **No frontend assumptions.** The engine is a library with a CLI; any
  frontend (React, Streamlit, notebook) can consume it via the JSON contract.

**Data flow:**

```
UserRequest
   │
   ▼
_resolve_universe    ── sector_map.indices_for_sectors
   │
   ▼
load_universe        ── data_loader (CSV → aligned returns)
   │
   ▼
estimators           ── james_stein_mean + ledoit_wolf_cov
   │
   ▼
run_selection        ── _pick_candidates → _dispatch → _feasibility → _score
   │
   ▼
PortfolioResult      ── weights + metrics + candidate table
```

---

## 15. Reference Cards

### 15.1 Every primary goal at a glance

| `primary_goal` | Model(s) used | Best for |
|---|---|---|
| `max_sharpe` | max_sharpe → risk_parity | Best risk-adjusted return |
| `max_return` | max_sharpe | Maximum return under diversification |
| `min_risk` | min_variance → risk_parity → max_sharpe | Lowest volatility with a return floor |
| `balanced` | risk_parity → hrp → max_diversification | Equally-diversified risk |
| `min_tail_risk` | min_cvar → min_variance → max_sharpe | Cap catastrophic losses |
| `min_drawdown` | min_max_drawdown → risk_parity → max_sharpe | Cap max peak-to-trough |
| `max_sortino` | max_sortino → max_sharpe | Reward asymmetric upside |
| `max_omega` | max_omega → max_sharpe | Probability-weighted upside/downside |
| `max_diversification` | max_diversification → risk_parity | Choueifaty DR |
| `inverse_vol` | inverse_vol → risk_parity | Simple baseline |

### 15.2 Every constraint

| Constraint | Effect | Which solver activates |
|---|---|---|
| `target_return` | Floor on annualized return | `min_variance_for_return`, `min_cvar_for_return` |
| `max_volatility` | Cap on annualized vol | `max_return_for_vol` |
| `max_drawdown` | Cap on realized max drawdown | `min_max_drawdown` |
| `max_cvar` | Cap on daily 5 % CVaR | `min_cvar_for_return` / `min_cvar` |
| `sectors` | Universe filter | *all* |
| `w_max`, `w_min` | Per-index weight bounds | *all* |
| `views` | Blended posterior | `black_litterman` |

### 15.3 CLI cheatsheet

```bash
python cli.py sectors                              # tag catalog
python cli.py indices --sector IT                  # indices per tag
python cli.py optimize <request.json>              # optimize once
python cli.py optimize <request.json> --out r.json # optimize + save
python cli.py backtest  <request.json> --rebalance Q --lookback 3
```

### 15.4 Sample minimal requests

```json
{"primary_goal": "max_sharpe"}
```

```json
{"target_return": 0.15, "max_volatility": 0.20}
```

```json
{"sectors": ["Banks", "IT"], "primary_goal": "balanced"}
```

```json
{"sectors": ["Broad"],
 "views": [{"asset": "nifty_it", "return": 0.20, "confidence": 0.6}]}
```

---

*Built for Macrowise. Data window: 2000-01-03 to 2026-07-10. 264 Indian
indices across BSE and NSE. Sources: yfinance, NSE API, BSE API (via `collect_indices.py`).*
