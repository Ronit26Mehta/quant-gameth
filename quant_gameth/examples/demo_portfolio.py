"""
demo_portfolio.py — Portfolio Optimization Showcase
====================================================
Run:  python -m quant_gameth.examples.demo_portfolio

Demonstrates:
  1. Synthetic market data generation
  2. Markowitz analytical solution
  3. Discrete portfolio selection (cardinality constraint)
  4. Simulated annealing portfolio
  5. Efficient frontier sweep
"""

from __future__ import annotations

import numpy as np


def main() -> None:
    print("=" * 70)
    print("  PORTFOLIO OPTIMIZATION DEMO — quant-gameth")
    print("=" * 70)

    from quant_gameth.generators.market import generate_portfolio_data
    from quant_gameth.solvers.portfolio import solve_portfolio

    # ------------------------------------------------------------------
    # 1. Generate synthetic market data
    # ------------------------------------------------------------------
    print("\n▸ 1) Generate market data (10 assets, 252 trading days)")
    mu, sigma = generate_portfolio_data(n_assets=10, seed=42)
    print(f"  Expected returns: {np.round(mu * 252, 4)}")  # annualised
    print(f"  Covariance shape: {sigma.shape}")
    print(f"  Return range: [{mu.min()*252:.4f}, {mu.max()*252:.4f}] annualised")

    # ------------------------------------------------------------------
    # 2. Markowitz analytical
    # ------------------------------------------------------------------
    print("\n▸ 2) Markowitz analytical solution (λ=1.0)")
    mark = solve_portfolio(mu, sigma, risk_aversion=1.0, method="markowitz")
    print(f"  Weights: {np.round(mark.solution, 4)}")
    print(f"  Expected return: {mark.metadata['expected_return']*252:.4f} (annualised)")
    print(f"  Risk (variance): {mark.metadata['risk']*252:.6f} (annualised)")
    print(f"  Sharpe ratio   : {mark.metadata['sharpe_ratio']:.4f}")
    n_active = np.sum(mark.solution > 0.01)
    print(f"  Active assets  : {int(n_active)}/10")

    # ------------------------------------------------------------------
    # 3. Discrete portfolio — pick best k=5 assets
    # ------------------------------------------------------------------
    print("\n▸ 3) Discrete portfolio (budget=5 assets)")
    disc = solve_portfolio(mu, sigma, risk_aversion=1.0, budget=5,
                           method="discrete")
    print(f"  Weights: {np.round(disc.solution, 4)}")
    print(f"  Expected return: {disc.metadata['expected_return']*252:.4f}")
    print(f"  Risk           : {disc.metadata['risk']*252:.6f}")
    selected = np.where(disc.solution > 0.01)[0]
    print(f"  Selected assets: {selected.tolist()}")

    # ------------------------------------------------------------------
    # 4. Simulated annealing
    # ------------------------------------------------------------------
    print("\n▸ 4) Simulated annealing portfolio (λ=1.0)")
    sa = solve_portfolio(mu, sigma, risk_aversion=1.0, method="annealing", seed=42)
    print(f"  Weights: {np.round(sa.solution, 4)}")
    print(f"  Expected return: {sa.metadata['expected_return']*252:.4f}")
    print(f"  Risk           : {sa.metadata['risk']*252:.6f}")
    print(f"  Time: {sa.time_seconds*1000:.1f}ms")

    # ------------------------------------------------------------------
    # 5. Risk aversion sweep (efficient frontier)
    # ------------------------------------------------------------------
    print("\n▸ 5) Efficient frontier sweep (λ = 0.1 to 10)")
    lambdas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    print(f"  {'λ':>6s}  {'Return':>10s}  {'Risk':>10s}  {'Sharpe':>8s}  {'#Assets':>8s}")
    print(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*8}")
    for lam in lambdas:
        r = solve_portfolio(mu, sigma, risk_aversion=lam, method="markowitz")
        n_act = int(np.sum(r.solution > 0.01))
        print(f"  {lam:6.1f}  {r.metadata['expected_return']*252:10.4f}  "
              f"{r.metadata['risk']*252:10.6f}  "
              f"{r.metadata['sharpe_ratio']:8.4f}  {n_act:8d}")

    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  ✓ Portfolio demo complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
