"""
Market/financial data generators.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def generate_portfolio_data(
    n_assets: int = 10,
    n_periods: int = 252,
    mean_return: float = 0.08,
    volatility: float = 0.20,
    correlation: float = 0.3,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic portfolio data.

    Returns
    -------
    expected_returns : np.ndarray, shape ``(n_assets,)``
    covariance_matrix : np.ndarray, shape ``(n_assets, n_assets)``
    """
    rng = np.random.default_rng(seed)

    # Expected returns with some dispersion
    mu = rng.normal(mean_return / n_periods, volatility / np.sqrt(n_periods), n_assets)

    # Covariance matrix with uniform correlation
    vols = rng.uniform(volatility * 0.5, volatility * 1.5, n_assets) / np.sqrt(n_periods)
    corr = np.full((n_assets, n_assets), correlation)
    np.fill_diagonal(corr, 1.0)

    # Ensure positive semi-definite
    D = np.diag(vols)
    sigma = D @ corr @ D

    # Add small regularisation
    sigma += np.eye(n_assets) * 1e-6

    return mu, sigma


def generate_auction_valuations(
    n_bidders: int = 5,
    n_items: int = 1,
    distribution: str = "uniform",
    seed: int = 42,
) -> np.ndarray:
    """Generate random auction valuations.

    Parameters
    ----------
    n_bidders : int
    n_items : int
    distribution : str
        ``'uniform'``, ``'normal'``, ``'exponential'``.
    seed : int

    Returns
    -------
    np.ndarray
        Shape ``(n_bidders,)`` for single item, ``(n_bidders, n_items)`` for multi-item.
    """
    rng = np.random.default_rng(seed)
    shape = (n_bidders,) if n_items == 1 else (n_bidders, n_items)

    if distribution == "normal":
        vals = np.abs(rng.normal(50, 15, shape))
    elif distribution == "exponential":
        vals = rng.exponential(50, shape)
    else:
        vals = rng.uniform(0, 100, shape)

    return vals
