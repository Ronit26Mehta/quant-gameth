"""
demo_trading.py — Auction & Market Mechanism Showcase
======================================================
Run:  python -m quant_gameth.examples.demo_trading

Demonstrates:
  1. First-price sealed-bid auction
  2. Second-price (Vickrey) auction
  3. VCG mechanism
  4. English ascending auction
  5. Revenue equivalence comparison
  6. Game-theoretic auction analysis
"""

from __future__ import annotations

import numpy as np


def main() -> None:
    print("=" * 70)
    print("  AUCTION & MARKET MECHANISM DEMO — quant-gameth")
    print("=" * 70)

    from quant_gameth.generators.market import generate_auction_valuations
    from quant_gameth.games.mechanism import (
        first_price_auction,
        second_price_auction,
        vcg_auction,
        english_auction,
        revenue_equivalence_demo,
    )

    # ------------------------------------------------------------------
    # 1. Generate bidder valuations
    # ------------------------------------------------------------------
    print("\n▸ 1) Generate bidder valuations (5 bidders, uniform)")
    valuations = generate_auction_valuations(n_bidders=5, seed=42)
    print(f"  Valuations: {np.round(valuations, 2)}")

    # ------------------------------------------------------------------
    # 2. First-price sealed-bid
    # ------------------------------------------------------------------
    print("\n▸ 2) First-price sealed-bid auction")
    fp = first_price_auction(valuations)
    print(f"  Winner  : Bidder {fp['winner']}")
    print(f"  Payment : {fp['payment']:.2f}")
    print(f"  Revenue : {fp['revenue']:.2f}")

    # ------------------------------------------------------------------
    # 3. Second-price (Vickrey)
    # ------------------------------------------------------------------
    print("\n▸ 3) Second-price (Vickrey) auction")
    sp = second_price_auction(valuations)
    print(f"  Winner  : Bidder {sp['winner']}")
    print(f"  Payment : {sp['payment']:.2f}  (second-highest bid)")
    print(f"  Revenue : {sp['revenue']:.2f}")
    print(f"  Note    : truthful bidding is dominant strategy")

    # ------------------------------------------------------------------
    # 4. VCG mechanism
    # ------------------------------------------------------------------
    print("\n▸ 4) VCG mechanism")
    vcg = vcg_auction(valuations)
    print(f"  Winner  : Bidder {vcg['winner']}")
    print(f"  Payment : {vcg['payment']:.2f}")
    print(f"  Social welfare: {vcg.get('social_welfare', 'N/A')}")

    # ------------------------------------------------------------------
    # 5. English ascending auction
    # ------------------------------------------------------------------
    print("\n▸ 5) English ascending auction")
    eng = english_auction(valuations)
    print(f"  Winner  : Bidder {eng['winner']}")
    print(f"  Payment : {eng['payment']:.2f}")
    print(f"  Rounds  : {eng.get('rounds', 'N/A')}")

    # ------------------------------------------------------------------
    # 6. Revenue equivalence
    # ------------------------------------------------------------------
    print("\n▸ 6) Revenue Equivalence Theorem (1000 simulations)")
    rev = revenue_equivalence_demo(n_bidders=5, n_simulations=1000, seed=42)
    print(f"  First-price avg revenue : {rev['first_price_revenue']:.2f}")
    print(f"  Second-price avg revenue: {rev['second_price_revenue']:.2f}")
    print(f"  Ratio (should ≈ 1.0)    : {rev.get('ratio', 0):.4f}")

    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  ✓ Trading & Auctions demo complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
