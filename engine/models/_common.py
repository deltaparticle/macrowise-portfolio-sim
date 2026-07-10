from __future__ import annotations
import numpy as np
import cvxpy as cp
from scipy.optimize import minimize


def base_constraints(w: cp.Variable, w_min: float = 0.0, w_max: float = 1.0, budget: float = 1.0):
    """Long-only, sum=budget, [w_min, w_max] per asset."""
    return [cp.sum(w) == budget, w >= w_min, w <= w_max]


def solve_cvx(prob: cp.Problem) -> str:
    for solver in ("ECOS", "SCS", "CLARABEL"):
        try:
            prob.solve(solver=solver, verbose=False)
            if prob.status in ("optimal", "optimal_inaccurate"):
                return prob.status
        except Exception:
            continue
    return prob.status or "failed"


def project_to_simplex(
    x: np.ndarray,
    w_min: float = 0.0,
    w_max: float = 1.0,
    budget: float = 1.0,
) -> np.ndarray:
    """Project an unconstrained vector onto the bounded simplex."""
    x = np.asarray(x, dtype=float).flatten()
    if x.size == 0:
        return x

    def objective(y: np.ndarray) -> float:
        return float(np.sum((y - x) ** 2))

    bounds = [(w_min, w_max)] * x.size
    constraints = [{"type": "eq", "fun": lambda y: np.sum(y) - budget}]
    start = np.clip(x, w_min, w_max)
    if start.sum() <= 0:
        start = np.full(x.size, budget / x.size)

    res = minimize(objective, start, bounds=bounds, constraints=constraints)
    if res.success and np.all(np.isfinite(res.x)):
        y = np.clip(np.asarray(res.x, dtype=float), w_min, w_max)
        if not np.isclose(y.sum(), budget):
            y = y / y.sum() * budget if y.sum() > 0 else np.full(x.size, budget / x.size)
        return np.clip(y, w_min, w_max)

    y = np.clip(start, w_min, w_max)
    if y.sum() <= 0:
        y = np.full(x.size, budget / x.size)
    return np.clip(y / y.sum() * budget if y.sum() > 0 else np.full(x.size, budget / x.size), w_min, w_max)


def clean_weights(
    w: np.ndarray,
    threshold: float = 1e-4,
    w_min: float = 0.0,
    w_max: float = 1.0,
    budget: float = 1.0,
) -> np.ndarray:
    w = np.asarray(w).flatten()
    w[np.abs(w) < threshold] = 0.0
    return project_to_simplex(w, w_min=w_min, w_max=w_max, budget=budget)
