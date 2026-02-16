"""
demo_games.py — Game Theory Engine Showcase
=============================================
Run:  python -m quant_gameth.examples.demo_games

Demonstrates:
  1. Prisoner's Dilemma — Nash equilibrium
  2. Battle of the Sexes — multiple equilibria
  3. Extensive-form game — backward induction
  4. Evolutionary dynamics — replicator for Hawk-Dove
  5. Cooperative games — Shapley value
  6. Quantum Prisoner's Dilemma — quantum advantage
"""

from __future__ import annotations

import numpy as np


def main() -> None:
    print("=" * 70)
    print("  GAME THEORY ENGINE DEMO — quant-gameth")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Prisoner's Dilemma — Nash Equilibrium
    # ------------------------------------------------------------------
    print("\n▸ 1) Prisoner's Dilemma — Nash Equilibrium")
    from quant_gameth.games.normal_form import NormalFormGame

    pd = NormalFormGame.prisoners_dilemma()
    equilibria = pd.find_nash()
    print(f"  Payoff matrices:")
    print(f"    Player 1:\n{pd.A}")
    print(f"    Player 2:\n{pd.B}")
    for eq in equilibria:
        print(f"  Nash equilibrium: P1={eq.strategies[0]}, P2={eq.strategies[1]}")
        print(f"    Payoffs: {eq.payoffs}")
        print(f"    Pure: {eq.is_pure}")

    # ------------------------------------------------------------------
    # 2. Battle of the Sexes — Multiple Equilibria
    # ------------------------------------------------------------------
    print("\n▸ 2) Battle of the Sexes — multiple equilibria")
    bos = NormalFormGame.battle_of_sexes()
    equilibria = bos.find_nash()
    print(f"  Found {len(equilibria)} Nash equilibria:")
    for i, eq in enumerate(equilibria):
        print(f"    NE {i+1}: P1={np.round(eq.strategies[0], 3)}, "
              f"P2={np.round(eq.strategies[1], 3)}, payoffs={np.round(eq.payoffs, 3)}")

    # ------------------------------------------------------------------
    # 3. Extensive-Form Game — Backward Induction
    # ------------------------------------------------------------------
    print("\n▸ 3) Ultimatum Game — backward induction")
    from quant_gameth.games.extensive_form import ExtensiveFormGame

    ult = ExtensiveFormGame.ultimatum(total=10)
    spe = ult.backward_induction()
    print(f"  Subgame-perfect equilibrium:")
    print(f"    Payoffs: {spe.payoffs}")

    # ------------------------------------------------------------------
    # 4. Evolutionary Dynamics — Hawk-Dove
    # ------------------------------------------------------------------
    print("\n▸ 4) Hawk-Dove replicator dynamics")
    from quant_gameth.games.evolutionary import replicator_dynamics

    # Hawk-Dove payoff matrix: V=4, C=6
    # H vs H: (V-C)/2 = -1, H vs D: V = 4, D vs H: 0, D vs D: V/2 = 2
    payoff = np.array([[-1, 4], [0, 2]], dtype=float)
    x0 = np.array([0.8, 0.2])  # start mostly hawks
    trajectory = replicator_dynamics(payoff, x0, dt=0.01, n_steps=500)
    final = trajectory[-1]
    print(f"  Initial : Hawk={x0[0]:.2f}, Dove={x0[1]:.2f}")
    print(f"  Final   : Hawk={final[0]:.4f}, Dove={final[1]:.4f}")
    print(f"  Theory  : Hawk=V/C={4/6:.4f}")

    # ------------------------------------------------------------------
    # 5. Cooperative Game — Shapley Value
    # ------------------------------------------------------------------
    print("\n▸ 5) Cooperative game — Shapley value")
    from quant_gameth.games.cooperative import shapley_value

    # Airport game: 3 planes, runway costs = [10, 20, 30]
    def airport_value(coalition: set) -> float:
        costs = {0: 10, 1: 20, 2: 30}
        if not coalition:
            return 0.0
        return max(costs[i] for i in coalition)

    sv = shapley_value(3, airport_value)
    print(f"  Shapley values: {[f'{v:.2f}' for v in sv]}")
    print(f"  Sum: {sum(sv):.2f}  (should equal v({{0,1,2}}) = 30)")

    # ------------------------------------------------------------------
    # 6. Quantum Prisoner's Dilemma
    # ------------------------------------------------------------------
    print("\n▸ 6) Quantum Prisoner's Dilemma — quantum advantage")
    from quant_gameth.games.quantum_games import quantum_prisoners_dilemma

    result = quantum_prisoners_dilemma()
    print(f"  Classical NE payoff: (1, 1)  [both defect]")
    print(f"  Quantum NE payoff  : ({result['quantum_payoff'][0]:.2f}, "
          f"{result['quantum_payoff'][1]:.2f})")
    print(f"  Quantum advantage  : {result.get('advantage', 'demonstrated')}")

    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  ✓ Game Theory demo complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
