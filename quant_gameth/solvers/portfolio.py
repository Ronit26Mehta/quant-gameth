"""
Portfolio optimization solver — Markowitz, QUBO, quantum-enhanced.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from quant_gameth._types import SolverResult, SolverMethod


def solve_portfolio(
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    risk_aversion: float = 1.0,
    budget: Optional[int] = None,
    method: str = "markowitz",
    seed: int = 42,
) -> SolverResult:
    """Solve portfolio optimization.

    Objective: max μᵀw - λ wᵀΣw  subject to 1ᵀw = 1, w ≥ 0

    Parameters
    ----------
    expected_returns : np.ndarray
        Expected returns, shape ``(n,)``.
    covariance_matrix : np.ndarray
        Covariance matrix, shape ``(n, n)``.
    risk_aversion : float
        Risk aversion parameter λ.
    budget : int or None
        Max number of assets to select (cardinality constraint).
    method : str
        ``'markowitz'``, ``'discrete'``, ``'annealing'``.
    seed : int
    """
    if method == "markowitz":
        return _markowitz_analytical(expected_returns, covariance_matrix, risk_aversion)
    elif method == "discrete" and budget is not None:
        return _discrete_portfolio(expected_returns, covariance_matrix,
                                   risk_aversion, budget, seed)
    else:
        return _portfolio_annealing(expected_returns, covariance_matrix,
                                    risk_aversion, budget, seed)


def _markowitz_analytical(
    mu: np.ndarray, sigma: np.ndarray, lam: float
) -> SolverResult:
    """Analytical Markowitz mean-variance solution."""
    t0 = time.perf_counter()
    n = len(mu)

    try:
        sigma_inv = np.linalg.inv(sigma + 1e-8 * np.eye(n))
    except np.linalg.LinAlgError:
        sigma_inv = np.linalg.pinv(sigma)

    ones = np.ones(n)

    # Optimal weights: w* = (1/2λ) Σ⁻¹ μ + ν Σ⁻¹ 1
    # where ν is chosen so that 1ᵀw* = 1
    raw_w = sigma_inv @ mu / (2 * lam)
    correction = sigma_inv @ ones
    nu = (1.0 - ones @ raw_w) / (ones @ correction)
    w = raw_w + nu * correction

    # Project to non-negative (simplified)
    w = np.maximum(w, 0)
    total = w.sum()
    if total > 0:
        w /= total

    portfolio_return = float(mu @ w)
    portfolio_risk = float(w @ sigma @ w)
    sharpe = portfolio_return / max(np.sqrt(portfolio_risk), 1e-14)

    return SolverResult(
        solution=w,
        energy=-(portfolio_return - lam * portfolio_risk),
        method=SolverMethod.ANALYTICAL,
        iterations=1,
        time_seconds=time.perf_counter() - t0,
        converged=True,
        metadata={
            "expected_return": portfolio_return,
            "risk": portfolio_risk,
            "sharpe_ratio": sharpe,
        },
    )


def _discrete_portfolio(
    mu: np.ndarray, sigma: np.ndarray, lam: float,
    budget: int, seed: int
) -> SolverResult:
    """Discrete portfolio: select exactly k assets."""
    from itertools import combinations

    t0 = time.perf_counter()
    n = len(mu)
    best_obj = -np.inf
    best_w = np.zeros(n)

    for combo in combinations(range(n), min(budget, n)):
        combo_list = list(combo)
        k = len(combo_list)
        mu_sub = mu[combo_list]
        sigma_sub = sigma[np.ix_(combo_list, combo_list)]

        try:
            sigma_inv = np.linalg.inv(sigma_sub + 1e-8 * np.eye(k))
        except np.linalg.LinAlgError:
            continue

        raw_w = sigma_inv @ mu_sub / (2 * lam)
        ones = np.ones(k)
        corr = sigma_inv @ ones
        nu = (1.0 - ones @ raw_w) / max(ones @ corr, 1e-14)
        w_sub = raw_w + nu * corr
        w_sub = np.maximum(w_sub, 0)
        total = w_sub.sum()
        if total > 0:
            w_sub /= total

        obj = float(mu_sub @ w_sub - lam * w_sub @ sigma_sub @ w_sub)
        if obj > best_obj:
            best_obj = obj
            best_w = np.zeros(n)
            for idx, ci in enumerate(combo_list):
                best_w[ci] = w_sub[idx]

    return SolverResult(
        solution=best_w,
        energy=-best_obj,
        method=SolverMethod.BRUTE_FORCE,
        iterations=1,
        time_seconds=time.perf_counter() - t0,
        converged=True,
        metadata={
            "expected_return": float(mu @ best_w),
            "risk": float(best_w @ sigma @ best_w),
            "n_assets": budget,
        },
    )


def _portfolio_annealing(
    mu: np.ndarray, sigma: np.ndarray, lam: float,
    budget: Optional[int], seed: int
) -> SolverResult:
    t0 = time.perf_counter()
    n = len(mu)
    rng = np.random.default_rng(seed)

    # Start from equal weights
    w = np.ones(n) / n
    best_w = w.copy()
    best_obj = float(mu @ w - lam * w @ sigma @ w)

    for step in range(5000):
        T = max(0.001, 1.0 * (1.0 - step / 5000))
        # Perturb
        new_w = w + rng.normal(0, 0.05, n)
        new_w = np.maximum(new_w, 0)
        total = new_w.sum()
        if total > 0:
            new_w /= total

        if budget is not None:
            # Keep only top-k
            top_k = np.argsort(-new_w)[:budget]
            mask = np.zeros(n, dtype=bool)
            mask[top_k] = True
            new_w[~mask] = 0
            total = new_w.sum()
            if total > 0:
                new_w /= total

        obj = float(mu @ new_w - lam * new_w @ sigma @ new_w)
        delta = obj - float(mu @ w - lam * w @ sigma @ w)

        if delta > 0 or rng.random() < np.exp(delta / T):
            w = new_w
            if obj > best_obj:
                best_obj = obj
                best_w = w.copy()

    return SolverResult(
        solution=best_w,
        energy=-best_obj,
        method=SolverMethod.SIMULATED_ANNEALING,
        iterations=5000,
        time_seconds=time.perf_counter() - t0,
        converged=True,
        metadata={
            "expected_return": float(mu @ best_w),
            "risk": float(best_w @ sigma @ best_w),
        },
    )
