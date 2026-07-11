"""Pydantic request/response models for the FastAPI layer.

These mirror engine.UserRequest and PortfolioResult but with stricter
validation, JSON-serializable types (no pandas.Series), and OpenAPI-friendly
descriptions so the frontend can generate types from /openapi.json.
"""
from __future__ import annotations
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field, ConfigDict


PrimaryGoal = Literal[
    "max_sharpe", "max_return", "min_risk", "balanced",
    "min_tail_risk", "min_drawdown", "max_sortino", "max_omega",
    "max_diversification", "inverse_vol", "black_litterman",
]


class AbsoluteView(BaseModel):
    """A view on a single asset's expected return."""
    model_config = ConfigDict(extra="forbid")

    asset: str = Field(..., description="Index slug, e.g. 'nifty_it'")
    return_: float = Field(..., alias="return",
                           description="Expected annualized return, e.g. 0.20 for 20%")
    confidence: float = Field(0.5, ge=0.001, le=0.999,
                              description="View confidence in (0, 1). Higher = more weight on the view.")


class RelativeView(BaseModel):
    """A view that one basket outperforms another by some amount."""
    model_config = ConfigDict(extra="forbid")

    long: list[str] = Field(..., min_length=1, description="Assets expected to outperform")
    short: list[str] = Field(default_factory=list, description="Assets expected to underperform")
    return_: float = Field(..., alias="return",
                           description="Expected excess return of long basket over short basket")
    confidence: float = Field(0.5, ge=0.001, le=0.999)


class OptimizeRequest(BaseModel):
    """User request for a portfolio optimization run."""

    # Universe
    sectors: Optional[list[str]] = Field(
        None, description="Sector/size/style/theme tag filter, e.g. ['IT','Banks']")
    universe: Optional[list[str]] = Field(
        None, description="Explicit index slugs (overrides `sectors`)")
    sector_match_all: bool = Field(
        False, description="If True, require an index to have ALL listed tags")

    # Targets / constraints (any combination)
    primary_goal: Optional[PrimaryGoal] = Field(
        None, description="Override the auto-selector with a specific goal")
    target_return: Optional[float] = Field(
        None, description="Minimum annualized return (hard constraint)")
    max_volatility: Optional[float] = Field(
        None, gt=0, description="Annualized volatility cap")
    max_drawdown: Optional[float] = Field(
        None, gt=0, description="Max drawdown cap (positive number)")
    max_cvar: Optional[float] = Field(
        None, gt=0, description="Max daily CVaR at `cvar_alpha` level")
    cvar_alpha: float = Field(0.05, gt=0, lt=1)

    # Weight bounds
    w_min: float = Field(0.0, ge=0.0, le=1.0)
    w_max: float = Field(1.0, ge=0.0, le=1.0)

    # Data window
    start: Optional[str] = Field(None, description="ISO date, e.g. '2020-01-01'")
    end: Optional[str] = None
    lookback_years: int = Field(5, ge=1, le=25)

    # Model knobs
    risk_free_rate: float = Field(0.065, ge=0.0, le=0.25)

    # Covariance estimation
    cov_method: Literal["auto", "ledoit_wolf", "robust_lw", "ewma"] = Field(
        "auto",
        description=(
            "'auto' = winsorized-LW in calm regimes, EWMA in stress. "
            "'ledoit_wolf' = plain LW (no fat-tail correction). "
            "'robust_lw' = always winsorized LW. "
            "'ewma' = always regime-tracking."))
    ewma_halflife: int = Field(63, ge=5, le=365,
                               description="EWMA halflife in trading days")
    regime_threshold: float = Field(1.3, ge=1.0, le=3.0,
                                    description="Vol ratio above this triggers EWMA in auto mode")
    clip_percentile: float = Field(1.0, ge=0.1, le=5.0,
                                   description="Winsorization clip level (%) for fat-tail robustness")

    # Black-Litterman
    views: list[dict] = Field(
        default_factory=list,
        description="List of AbsoluteView or RelativeView dicts. Presence forces Black-Litterman.")
    bl_tau: float = Field(0.05, gt=0, le=1)

    # Forward simulation (bootstrap). Runs by default; set simulate=false to
    # skip. Output is dropped only if the sim crashes or exceeds sim_timeout_s.
    simulate: bool = Field(True,
                           description="If true (default), appends a Monte Carlo simulation of horizon-end outcomes.")
    horizon_years: float = Field(5.0, gt=0, le=30,
                                 description="Investment horizon in years for the simulation")
    n_simulations: int = Field(1000, ge=100, le=10000,
                               description="Number of bootstrap paths")
    target_total_return: Optional[float] = Field(
        None,
        description="Optional total-return target (e.g. 0.40 for 40%); reports prob_above_target.")
    sim_seed: int = Field(42, description="RNG seed for reproducibility")
    sim_timeout_s: float = Field(5.0, gt=0, le=60,
                                 description="Wall-clock cap on simulation. On timeout, sim is dropped (200 still returned).")

    # Output
    top_k: int = Field(3, ge=1, le=10)

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "sectors": ["IT", "Banks", "Auto", "Pharma", "FMCG"],
                    "primary_goal": "max_sharpe",
                    "w_max": 0.30,
                    "lookback_years": 5,
                },
                {
                    "sectors": ["Largecap", "Broad"],
                    "target_return": 0.15,
                    "max_volatility": 0.20,
                    "max_drawdown": 0.25,
                    "w_max": 0.20,
                },
                {
                    "sectors": ["IT", "Banks", "Auto", "Pharma"],
                    "views": [
                        {"asset": "nifty_it", "return": 0.20, "confidence": 0.65},
                        {"long": ["nifty_bank"], "short": ["nifty_pharma"],
                         "return": 0.04, "confidence": 0.5},
                    ],
                    "w_max": 0.30,
                },
            ]
        }
    )


class BacktestRequest(BaseModel):
    """Walk-forward backtest request."""
    model_config = ConfigDict(extra="forbid")

    request: OptimizeRequest
    rebalance: Literal["W", "M", "Q", "A"] = Field(
        "Q", description="Rebalance frequency: Weekly / Monthly / Quarterly / Annual")
    lookback_years: int = Field(3, ge=1, le=15,
                                description="Rolling estimation window at each rebalance")
    initial_capital: float = Field(100.0, gt=0)


class CandidateSummary(BaseModel):
    model: Optional[str] = None
    status: Optional[str] = None
    feasible: bool = False
    reason: Optional[str] = None
    score: Optional[float] = None
    metrics: dict[str, float] = Field(default_factory=dict)


class PortfolioMetrics(BaseModel):
    ann_return: float
    ann_vol: float
    sharpe: float
    sortino: float
    max_drawdown: float
    cvar: float
    calmar: float


class SimulationPercentiles(BaseModel):
    p10: float
    p25: float
    median: float
    p75: float
    p90: float
    mean: float
    std: float


class SimulationSummary(BaseModel):
    horizon_years: float
    n_simulations: int
    target_total_return: Optional[float] = None
    total_return: SimulationPercentiles
    annualized_return: SimulationPercentiles
    prob_above_target: Optional[float] = None
    method: str = "iid_bootstrap"
    note: str = ""


class OptimizeResponse(BaseModel):
    chosen_model: str
    feasible: bool
    reason: str = ""
    universe_size: int
    weights: dict[str, float]
    metrics: PortfolioMetrics
    candidates: list[CandidateSummary]
    universe: list[str]
    cov_method_used: str = "ledoit_wolf"
    vol_regime_ratio: float = 1.0
    simulation: Optional[SimulationSummary] = None


class BacktestResponse(BaseModel):
    rebalance: str
    lookback_years: int
    n_rebalances: int
    metrics: PortfolioMetrics
    equity_curve: list[dict[str, Any]]  # [{"date": "YYYY-MM-DD", "value": 123.4}, ...]
    rebalance_log: list[dict[str, Any]]


class SectorsResponse(BaseModel):
    sectors: list[str]
    sizes: list[str]
    styles: list[str]
    themes: list[str]


class IndicesResponse(BaseModel):
    sector: str
    match_any: bool
    count: int
    indices: list[str]


class ModelInfo(BaseModel):
    name: str
    family: str
    solver: str
    handles_target_return: bool
    handles_max_vol: bool
    handles_max_dd: bool
    handles_max_cvar: bool
    handles_views: bool
    description: str


class ModelsResponse(BaseModel):
    count: int
    models: list[ModelInfo]


class ConvertReturnResponse(BaseModel):
    total_return: float = Field(description="The total return input, e.g. 0.40 for 40%")
    years: int = Field(description="Investment horizon in years")
    annualized_return: float = Field(description="Equivalent annualized return (CAGR)")
    note: str = Field(description="Explanation of what this number means")


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str
    data_dir_present: bool
    n_indices_available: int


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
