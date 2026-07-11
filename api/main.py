"""FastAPI backend for the Macrowise portfolio optimization engine.

Run locally:
    uvicorn api.main:app --host 0.0.0.0 --port 8000

OpenAPI UI:
    http://localhost:8000/docs

Design choices:
- Sync endpoints. Longest optimizer (min_max_drawdown DE) runs in <3s
  after DE tuning; keep-alive is wide for long backtest walks.
- No auth for v1. Deploy behind Macrowise's gateway.
- CORS permissive; tighten allowed_origins for production.
- All engine errors are translated to HTTP 400/422 with a JSON payload the
  frontend can render directly.
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from engine.optimizer import UserRequest, optimize
from engine.backtest import walk_forward
from engine.sector_map import (
    available_sectors,
    build_map,
    indices_for_sectors,
)
from engine.config import DATA_DIR

from . import schemas as S

log = logging.getLogger("macrowise.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

VERSION = "0.1.0"

app = FastAPI(
    title="Macrowise Portfolio Optimization API",
    version=VERSION,
    description=(
        "REST wrapper around the Macrowise portfolio optimization engine. "
        "Supports 10 optimization models (Mean-Variance family, CVaR, Max Drawdown, "
        "Risk Parity, HRP, Black-Litterman, Sortino, Omega, Max Diversification, "
        "Inverse Vol). Universe = 264 Indian indices. See /docs for endpoints."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten for production
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _to_user_request(body: S.OptimizeRequest) -> UserRequest:
    """Pydantic model -> engine dataclass."""
    d = body.model_dump()
    # Rename 'return_' back to 'return' inside views if user used the alias form.
    d["views"] = [{("return" if k == "return_" else k): v for k, v in v.items()}
                  for v in d.get("views", [])]
    # engine.UserRequest doesn't accept market_weights via API; frontend can't send pd.Series.
    return UserRequest(**d)


def _metrics_payload(metrics: dict[str, Any]) -> S.PortfolioMetrics:
    return S.PortfolioMetrics(
        ann_return=float(metrics.get("ann_return", 0.0)),
        ann_vol=float(metrics.get("ann_vol", 0.0)),
        sharpe=float(metrics.get("sharpe", 0.0)),
        sortino=float(metrics.get("sortino", 0.0)),
        max_drawdown=float(metrics.get("max_drawdown", 0.0)),
        cvar=float(metrics.get("cvar", 0.0)),
        calmar=float(metrics.get("calmar", 0.0)),
    )


# ---------------------------------------------------------------------------
# error handling
# ---------------------------------------------------------------------------

@app.exception_handler(ValueError)
async def value_error_handler(request, exc: ValueError):
    return JSONResponse(status_code=400,
                        content={"error": "bad_request", "detail": str(exc)})


@app.exception_handler(RuntimeError)
async def runtime_error_handler(request, exc: RuntimeError):
    return JSONResponse(status_code=500,
                        content={"error": "engine_error", "detail": str(exc)})


# ---------------------------------------------------------------------------
# endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["meta"])
def root() -> dict[str, Any]:
    return {
        "service": "macrowise-portfolio-api",
        "version": VERSION,
        "docs": "/docs",
        "openapi": "/openapi.json",
        "endpoints": [
            "GET  /health",
            "POST /optimize",
            "POST /backtest",
            "GET  /sectors",
            "GET  /indices?sector=IT",
            "GET  /models",
            "GET  /examples",
        ],
    }


@app.get("/health", response_model=S.HealthResponse, tags=["meta"])
def health() -> S.HealthResponse:
    data_ok = DATA_DIR.exists()
    n_indices = len(list(DATA_DIR.glob("*_yfinance.csv"))) if data_ok else 0
    return S.HealthResponse(
        status="ok",
        version=VERSION,
        data_dir_present=data_ok,
        n_indices_available=n_indices,
    )


@app.post("/optimize", response_model=S.OptimizeResponse, tags=["optimize"])
def optimize_endpoint(body: S.OptimizeRequest) -> S.OptimizeResponse:
    """Run a portfolio optimization.

    The engine auto-selects the best-fitting model from the request's
    constraint mix. Returns weights, metrics, feasibility, and the full
    ranked candidate table.
    """
    req = _to_user_request(body)
    log.info("optimize: goal=%s sectors=%s targets=(ret=%s vol=%s dd=%s cvar=%s) views=%d",
             req.primary_goal, req.sectors, req.target_return, req.max_volatility,
             req.max_drawdown, req.max_cvar, len(req.views or []))
    result = optimize(req)

    return S.OptimizeResponse(
        chosen_model=result.chosen_model,
        feasible=bool(result.feasible),
        reason=result.reason or "",
        universe_size=result.n_assets_used,
        weights=result.weights,
        metrics=_metrics_payload(result.metrics),
        candidates=[
            S.CandidateSummary(
                model=c.get("model"),
                status=c.get("status"),
                feasible=bool(c.get("feasible", False)),
                reason=c.get("reason") or "",
                score=c.get("score"),
                metrics={k: float(v) for k, v in (c.get("metrics") or {}).items()},
            )
            for c in result.all_candidates
        ],
        universe=result.universe,
        cov_method_used=result.cov_method_used,
        vol_regime_ratio=float(result.vol_regime_ratio),
    )


@app.post("/backtest", response_model=S.BacktestResponse, tags=["optimize"])
def backtest_endpoint(body: S.BacktestRequest) -> S.BacktestResponse:
    """Walk-forward backtest: re-optimize at each rebalance date using a
    rolling estimation window. Returns equity curve + realized metrics."""
    req = _to_user_request(body.request)
    log.info("backtest: rebalance=%s lookback=%d goal=%s",
             body.rebalance, body.lookback_years, req.primary_goal)
    result = walk_forward(
        req,
        rebalance=body.rebalance,
        lookback_years=body.lookback_years,
        initial_capital=body.initial_capital,
    )
    curve = result["equity_curve"]
    equity = [{"date": pd.Timestamp(d).strftime("%Y-%m-%d"), "value": float(v)}
              for d, v in curve.items()]
    rlog = []
    for entry in result.get("rebalance_log", []):
        rlog.append({
            "date": pd.Timestamp(entry["date"]).strftime("%Y-%m-%d") if "date" in entry else None,
            "model": entry.get("model"),
            "weights": entry.get("weights", {}),
            "error": entry.get("error"),
        })
    return S.BacktestResponse(
        rebalance=body.rebalance,
        lookback_years=body.lookback_years,
        n_rebalances=len(result.get("rebalance_dates", [])),
        metrics=_metrics_payload(result["metrics"]),
        equity_curve=equity,
        rebalance_log=rlog,
    )


@app.get("/sectors", response_model=S.SectorsResponse, tags=["catalog"])
def sectors_endpoint() -> S.SectorsResponse:
    """Return the tag catalog: available sectors, sizes, styles, themes."""
    tags = available_sectors()
    return S.SectorsResponse(**tags)


@app.get("/indices", response_model=S.IndicesResponse, tags=["catalog"])
def indices_endpoint(
    sector: str = Query(..., description="Sector / size / style / theme tag, e.g. 'IT'"),
    match_any: bool = Query(True, description="Ignored for single tag; kept for API symmetry"),
) -> S.IndicesResponse:
    """List all indices tagged with a given sector/size/style/theme tag."""
    picks = indices_for_sectors([sector], match_any=match_any)
    return S.IndicesResponse(
        sector=sector, match_any=match_any, count=len(picks), indices=picks,
    )


@app.get("/models", response_model=S.ModelsResponse, tags=["catalog"])
def models_endpoint() -> S.ModelsResponse:
    """Metadata about the 10 optimization models available."""
    catalog = [
        S.ModelInfo(name="max_sharpe", family="mean_variance", solver="cvxpy (QP/SOCP)",
                    handles_target_return=False, handles_max_vol=False,
                    handles_max_dd=False, handles_max_cvar=False, handles_views=False,
                    description="Convex reformulation of the tangency portfolio."),
        S.ModelInfo(name="min_variance", family="mean_variance", solver="cvxpy (QP)",
                    handles_target_return=False, handles_max_vol=False,
                    handles_max_dd=False, handles_max_cvar=False, handles_views=False,
                    description="Global minimum variance portfolio."),
        S.ModelInfo(name="min_variance_for_return", family="mean_variance", solver="cvxpy (QP)",
                    handles_target_return=True, handles_max_vol=False,
                    handles_max_dd=False, handles_max_cvar=False, handles_views=False,
                    description="Min variance subject to a hard return floor."),
        S.ModelInfo(name="max_return_for_vol", family="mean_variance", solver="cvxpy (SOCP)",
                    handles_target_return=False, handles_max_vol=True,
                    handles_max_dd=False, handles_max_cvar=False, handles_views=False,
                    description="Max return subject to a volatility cap."),
        S.ModelInfo(name="min_cvar", family="cvar", solver="cvxpy (LP, Rockafellar-Uryasev)",
                    handles_target_return=False, handles_max_vol=False,
                    handles_max_dd=False, handles_max_cvar=True, handles_views=False,
                    description="Minimize daily CVaR at the given alpha level."),
        S.ModelInfo(name="min_cvar_for_return", family="cvar", solver="cvxpy (LP)",
                    handles_target_return=True, handles_max_vol=False,
                    handles_max_dd=False, handles_max_cvar=True, handles_views=False,
                    description="Min CVaR subject to a return floor."),
        S.ModelInfo(name="min_max_drawdown", family="drawdown", solver="scipy differential_evolution",
                    handles_target_return=True, handles_max_vol=False,
                    handles_max_dd=True, handles_max_cvar=False, handles_views=False,
                    description="Minimize realized max drawdown; non-convex, slow (~40s)."),
        S.ModelInfo(name="risk_parity", family="risk_parity", solver="scipy SLSQP",
                    handles_target_return=False, handles_max_vol=False,
                    handles_max_dd=False, handles_max_cvar=False, handles_views=False,
                    description="Equal Risk Contribution. No return estimate needed."),
        S.ModelInfo(name="hrp", family="risk_parity", solver="scipy hierarchical clustering",
                    handles_target_return=False, handles_max_vol=False,
                    handles_max_dd=False, handles_max_cvar=False, handles_views=False,
                    description="Lopez de Prado Hierarchical Risk Parity."),
        S.ModelInfo(name="black_litterman", family="bayesian",
                    solver="closed-form posterior + cvxpy max_sharpe",
                    handles_target_return=True, handles_max_vol=False,
                    handles_max_dd=False, handles_max_cvar=False, handles_views=True,
                    description="Idzorek-style Black-Litterman posterior."),
        S.ModelInfo(name="max_sortino", family="downside", solver="scipy SLSQP multi-start",
                    handles_target_return=False, handles_max_vol=False,
                    handles_max_dd=False, handles_max_cvar=False, handles_views=False,
                    description="Max Sortino ratio; downside-only risk."),
        S.ModelInfo(name="max_omega", family="downside", solver="scipy SLSQP multi-start",
                    handles_target_return=False, handles_max_vol=False,
                    handles_max_dd=False, handles_max_cvar=False, handles_views=False,
                    description="Max Omega ratio."),
        S.ModelInfo(name="max_diversification", family="diversification", solver="scipy SLSQP",
                    handles_target_return=False, handles_max_vol=False,
                    handles_max_dd=False, handles_max_cvar=False, handles_views=False,
                    description="Choueifaty diversification ratio."),
        S.ModelInfo(name="inverse_vol", family="baseline", solver="closed-form",
                    handles_target_return=False, handles_max_vol=False,
                    handles_max_dd=False, handles_max_cvar=False, handles_views=False,
                    description="Weights proportional to 1/sigma. Baseline."),
    ]
    return S.ModelsResponse(count=len(catalog), models=catalog)


EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


@app.get("/examples", tags=["catalog"])
def examples_endpoint() -> dict[str, Any]:
    """Return every example request bundled with the repo."""
    out = {}
    if not EXAMPLES_DIR.exists():
        return {"examples": out}
    for path in sorted(EXAMPLES_DIR.glob("*.json")):
        try:
            out[path.stem] = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:  # pragma: no cover
            out[path.stem] = {"error": f"cannot parse: {e}"}
    return {"count": len(out), "examples": out}
