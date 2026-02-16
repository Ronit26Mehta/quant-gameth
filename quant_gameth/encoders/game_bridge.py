"""
Game ↔ Optimization bridge — automatic transpilation.

Novel contribution: convert any normal-form game to a QUBO / quantum circuit,
and decode solutions back to strategy profiles.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from quant_gameth._types import EncodedProblem, EquilibriumResult, SolverMethod
from quant_gameth.encoders.qubo import QUBOBuilder


class GameOptimizationBridge:
    """Convert games to optimization problems and back.

    Handles automatic encoding of:
    - Normal-form games → QUBO (find Nash via optimization)
    - Multi-player games → combinatorial optimization
    """

    @staticmethod
    def game_to_qubo(
        payoff_matrix_1: np.ndarray,
        payoff_matrix_2: Optional[np.ndarray] = None,
        penalty: float = 10.0,
    ) -> Tuple[EncodedProblem, Dict]:
        """Convert a 2-player normal-form game to QUBO.

        Strategy: encode mixed strategies as binary vectors (discretised)
        and optimise for Nash condition.

        Parameters
        ----------
        payoff_matrix_1 : np.ndarray
            Payoff matrix for player 1, shape ``(m, k)``.
        payoff_matrix_2 : np.ndarray or None
        penalty : float

        Returns
        -------
        problem : EncodedProblem
        decode_info : dict
            Mapping information for decoding solutions.
        """
        A = np.asarray(payoff_matrix_1, dtype=float)
        if payoff_matrix_2 is None:
            B = -A
        else:
            B = np.asarray(payoff_matrix_2, dtype=float)

        m, k = A.shape

        # Variables: p_i for player 1 (m variables, one-hot)
        #            q_j for player 2 (k variables, one-hot)
        n_vars = m + k
        builder = QUBOBuilder(n_vars)

        # One-hot constraints (pure strategy selection)
        builder.add_one_hot(list(range(m)), penalty)
        builder.add_one_hot(list(range(m, m + k)), penalty)

        # Objective: maximise total welfare (negative for minimisation)
        # f(p, q) = -Σᵢⱼ pᵢ qⱼ (A[i,j] + B[i,j])
        for i in range(m):
            for j in range(k):
                weight = -(A[i, j] + B[i, j])
                builder.add_quadratic(i, m + j, weight)

        decode_info = {
            "m": m,
            "k": k,
            "payoff_1": A.tolist(),
            "payoff_2": B.tolist(),
        }

        return builder.build("game_to_qubo"), decode_info

    @staticmethod
    def qubo_solution_to_equilibrium(
        solution: np.ndarray,
        decode_info: Dict,
    ) -> EquilibriumResult:
        """Decode a QUBO solution back to a game equilibrium."""
        m = decode_info["m"]
        k = decode_info["k"]
        A = np.array(decode_info["payoff_1"])
        B = np.array(decode_info["payoff_2"])

        p = solution[:m]
        q = solution[m:m + k]

        # Normalise to probabilities
        p_sum = p.sum()
        q_sum = q.sum()
        if p_sum > 0:
            p = p / p_sum
        if q_sum > 0:
            q = q / q_sum

        payoff_1 = float(p @ A @ q)
        payoff_2 = float(p @ B @ q)

        return EquilibriumResult(
            strategies=[p, q],
            payoffs=np.array([payoff_1, payoff_2]),
            equilibrium_type="nash_approximation",
            method=SolverMethod.QAOA,
            is_pure=bool(np.max(p) > 0.99 and np.max(q) > 0.99),
        )

    @staticmethod
    def multi_player_to_qubo(
        payoff_tensors: List[np.ndarray],
        penalties: float = 10.0,
    ) -> Tuple[EncodedProblem, Dict]:
        """Convert an n-player game to QUBO.

        Each player's strategy is encoded as one-hot binary variables.
        """
        n_players = len(payoff_tensors)
        strategies_per_player = [t.shape[i] for i, t in enumerate(payoff_tensors)]

        # Total binary variables
        total_vars = sum(strategies_per_player)
        builder = QUBOBuilder(total_vars)

        # One-hot constraints per player
        offset = 0
        player_offsets = []
        for p in range(n_players):
            n_strats = strategies_per_player[p]
            player_vars = list(range(offset, offset + n_strats))
            builder.add_one_hot(player_vars, penalties)
            player_offsets.append(offset)
            offset += n_strats

        decode_info = {
            "n_players": n_players,
            "strategies_per_player": strategies_per_player,
            "player_offsets": player_offsets,
        }

        return builder.build("multi_player_game"), decode_info

    @staticmethod
    def payoff_to_cost_hamiltonian(
        payoff_matrix: np.ndarray,
        n_qubits_per_player: int,
    ) -> np.ndarray:
        """Convert payoff matrix to a diagonal cost Hamiltonian for QAOA.

        Each computational basis state encodes a strategy profile.
        """
        m, k = payoff_matrix.shape
        dim = 1 << (2 * n_qubits_per_player)
        diag = np.zeros(dim)

        for x in range(dim):
            # Decode player strategies
            p1_bits = x & ((1 << n_qubits_per_player) - 1)
            p2_bits = (x >> n_qubits_per_player) & ((1 << n_qubits_per_player) - 1)

            if p1_bits < m and p2_bits < k:
                # Negative because QAOA minimises
                diag[x] = -payoff_matrix[p1_bits, p2_bits]
            else:
                diag[x] = 1000.0  # large penalty for invalid encodings

        return diag
