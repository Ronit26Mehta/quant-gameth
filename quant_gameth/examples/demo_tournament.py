"""
demo_tournament.py — Multi-Strategy Tournament Showcase
========================================================
Run:  python -m quant_gameth.examples.demo_tournament

Demonstrates:
  1. Repeated Prisoner's Dilemma tournament
  2. Strategy comparison (TFT, Grim, Pavlov, Random, etc.)
  3. Round-robin ranking
  4. Evolutionary dynamics on tournament payoffs
"""

from __future__ import annotations

import numpy as np


def main() -> None:
    print("=" * 70)
    print("  STRATEGY TOURNAMENT DEMO — quant-gameth")
    print("=" * 70)

    from quant_gameth.games.normal_form import NormalFormGame
    from quant_gameth.games.repeated import (
        round_robin_tournament,
        iterated_game,
        BUILTIN_STRATEGIES,
    )

    # ------------------------------------------------------------------
    # 1. Setup
    # ------------------------------------------------------------------
    print("\n▸ 1) Setup: Prisoner's Dilemma iterated tournament")
    pd = NormalFormGame.prisoners_dilemma()
    print(f"  Payoff matrix (Player 1):")
    print(f"               Cooperate  Defect")
    print(f"    Cooperate     {pd.A[0,0]:.0f}        {pd.A[0,1]:.0f}")
    print(f"    Defect        {pd.A[1,0]:.0f}        {pd.A[1,1]:.0f}")

    strategies = list(BUILTIN_STRATEGIES.keys())
    print(f"\n  Competing strategies ({len(strategies)}):")
    for s in strategies:
        print(f"    • {s}")

    # ------------------------------------------------------------------
    # 2. Round-robin tournament
    # ------------------------------------------------------------------
    print("\n▸ 2) Round-robin tournament (200 rounds per match, δ=0.99)")
    ranking = round_robin_tournament(
        pd, strategies=strategies, n_rounds=200,
        discount=0.99, seed=42,
    )

    print(f"\n  {'Rank':>4s}  {'Strategy':<25s}  {'Total Score':>12s}")
    print(f"  {'─'*4}  {'─'*25}  {'─'*12}")
    for rank, (name, score) in enumerate(ranking, 1):
        bar = "█" * int(score / max(r[1] for r in ranking) * 20)
        print(f"  {rank:4d}  {name:<25s}  {score:12.1f}  {bar}")

    # ------------------------------------------------------------------
    # 3. Head-to-head: Tit-for-Tat vs Grim Trigger
    # ------------------------------------------------------------------
    print("\n▸ 3) Head-to-head: tit_for_tat vs grim_trigger (100 rounds)")
    h2h = iterated_game(
        pd,
        strategy_1="tit_for_tat",
        strategy_2="grim_trigger",
        n_rounds=100,
        discount=1.0,
        seed=42,
    )
    print(f"  TFT total   : {h2h['score_1']:.0f}")
    print(f"  Grim total  : {h2h['score_2']:.0f}")
    print(f"  Cooperation rate: {h2h.get('cooperation_rate', 'N/A')}")

    # ------------------------------------------------------------------
    # 4. Head-to-head: Tit-for-Tat vs Always Defect
    # ------------------------------------------------------------------
    print("\n▸ 4) Head-to-head: tit_for_tat vs always_defect (100 rounds)")
    h2h2 = iterated_game(
        pd,
        strategy_1="tit_for_tat",
        strategy_2="always_defect",
        n_rounds=100,
        discount=1.0,
        seed=42,
    )
    print(f"  TFT total        : {h2h2['score_1']:.0f}")
    print(f"  Always-D total   : {h2h2['score_2']:.0f}")

    # ------------------------------------------------------------------
    # 5. Evolutionary dynamics from tournament
    # ------------------------------------------------------------------
    print("\n▸ 5) Evolutionary dynamics (replicator on tournament payoffs)")
    from quant_gameth.games.evolutionary import replicator_dynamics

    # Build payoff matrix from average tournament scores
    n_strats = len(strategies)
    payoff_matrix = np.zeros((n_strats, n_strats))
    for i, s1 in enumerate(strategies):
        for j, s2 in enumerate(strategies):
            if i == j:
                payoff_matrix[i, j] = 3.0 * 200  # cooperate with self-copy
            else:
                h = iterated_game(pd, strategy_1=s1, strategy_2=s2,
                                  n_rounds=200, discount=1.0, seed=42)
                payoff_matrix[i, j] = h["score_1"]

    # Start with uniform population
    x0 = np.ones(n_strats) / n_strats
    trajectory = replicator_dynamics(payoff_matrix, x0, dt=0.001, n_steps=2000)
    final = trajectory[-1]

    print(f"\n  Final population shares:")
    for i, s in enumerate(strategies):
        if final[i] > 0.01:
            bar = "█" * int(final[i] * 40)
            print(f"    {s:<25s} {final[i]:6.3f}  {bar}")

    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  ✓ Tournament demo complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
