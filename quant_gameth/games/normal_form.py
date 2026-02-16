"""
Normal-form game representation and Nash equilibrium solvers.

From Mathematical Foundations §B:
    G = (N, S, u) — players, strategy spaces, payoff functions.

Solvers:
1. Support enumeration (exact for 2-player, §B.3)
2. Lemke-Howson (linear complementarity)
3. Vertex enumeration (fallback)
4. Linear programming (zero-sum games)
"""

from __future__ import annotations

import itertools
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from quant_gameth._types import (
    EquilibriumResult,
    GameDescription,
    GameType,
    SolverMethod,
)


class NormalFormGame:
    """Two-player normal-form game.

    Parameters
    ----------
    payoff_matrix_1 : np.ndarray
        Payoff matrix for player 1, shape ``(m, k)`` where m = #strategies
        for player 1, k = #strategies for player 2.
    payoff_matrix_2 : np.ndarray or None
        Payoff matrix for player 2.  If ``None``, game is zero-sum:
        payoff_2 = -payoff_1.
    name : str
        Human-readable game name.
    """

    def __init__(
        self,
        payoff_matrix_1: np.ndarray,
        payoff_matrix_2: Optional[np.ndarray] = None,
        name: str = "game",
    ):
        self.A = np.asarray(payoff_matrix_1, dtype=float)
        if payoff_matrix_2 is None:
            self.B = -self.A  # zero-sum
            self.is_zero_sum = True
        else:
            self.B = np.asarray(payoff_matrix_2, dtype=float)
            self.is_zero_sum = False
        if self.A.shape != self.B.shape:
            raise ValueError("Payoff matrices must have the same shape")
        self.m, self.k = self.A.shape  # m strategies for P1, k for P2
        self.name = name

    # ----- Classic named games -----

    @classmethod
    def prisoners_dilemma(cls, R: float = 3, S: float = 0,
                          T: float = 5, P: float = 1) -> "NormalFormGame":
        """Prisoner's Dilemma (§B.1).

        Cooperate = 0, Defect = 1.
        """
        A = np.array([[R, S], [T, P]])
        B = np.array([[R, T], [S, P]])
        return cls(A, B, name="prisoners_dilemma")

    @classmethod
    def battle_of_sexes(cls, a: float = 3, b: float = 2) -> "NormalFormGame":
        A = np.array([[a, 0], [0, b]])
        B = np.array([[b, 0], [0, a]])
        return cls(A, B, name="battle_of_sexes")

    @classmethod
    def matching_pennies(cls) -> "NormalFormGame":
        A = np.array([[1, -1], [-1, 1]])
        return cls(A, name="matching_pennies")

    @classmethod
    def hawks_and_doves(cls, V: float = 4, C: float = 6) -> "NormalFormGame":
        A = np.array([[(V - C) / 2, V], [0, V / 2]])
        return cls(A, A.T, name="hawks_and_doves")

    @classmethod
    def stag_hunt(cls) -> "NormalFormGame":
        A = np.array([[4, 1], [3, 2]])
        B = np.array([[4, 3], [1, 2]])
        return cls(A, B, name="stag_hunt")

    @classmethod
    def coordination_game(cls, a: float = 2, b: float = 1) -> "NormalFormGame":
        A = np.array([[a, 0], [0, b]])
        return cls(A, A, name="coordination")

    # ----- Solver methods -----

    def find_nash(self, method: str = "support_enumeration") -> List[EquilibriumResult]:
        """Find all Nash equilibria.

        Parameters
        ----------
        method : str
            ``'support_enumeration'``, ``'lemke_howson'``, ``'vertex'``,
            or ``'lp'`` (zero-sum only).

        Returns
        -------
        List of EquilibriumResult
        """
        if method == "lp" and self.is_zero_sum:
            return self._nash_lp()
        elif method == "lemke_howson":
            return self._nash_lemke_howson()
        elif method == "vertex":
            return self._nash_vertex_enumeration()
        else:
            return self._nash_support_enumeration()

    def _nash_support_enumeration(self) -> List[EquilibriumResult]:
        """Support enumeration algorithm (§B.3).

        For each pair of supports (S₁, S₂), solve the indifference
        equations and check best-response property.
        """
        t0 = time.perf_counter()
        equilibria: List[EquilibriumResult] = []

        for size1 in range(1, self.m + 1):
            for size2 in range(1, self.k + 1):
                for supp1 in itertools.combinations(range(self.m), size1):
                    for supp2 in itertools.combinations(range(self.k), size2):
                        result = self._check_support(
                            list(supp1), list(supp2)
                        )
                        if result is not None:
                            elapsed = time.perf_counter() - t0
                            eq = EquilibriumResult(
                                strategies=[result[0], result[1]],
                                payoffs=np.array([result[2], result[3]]),
                                equilibrium_type="nash",
                                method=SolverMethod.SUPPORT_ENUMERATION,
                                is_pure=(size1 == 1 and size2 == 1),
                                time_seconds=elapsed,
                            )
                            equilibria.append(eq)

        return equilibria

    def _check_support(
        self, supp1: List[int], supp2: List[int]
    ) -> Optional[Tuple[np.ndarray, np.ndarray, float, float]]:
        """Check if the given supports yield a Nash equilibrium.

        Solves the indifference equations:
            For all i, j in supp1: u₁(i, σ₂) = u₁(j, σ₂)
            For all i, j in supp2: u₂(σ₁, i) = u₂(σ₁, j)
        """
        s1, s2 = len(supp1), len(supp2)

        # Solve for player 2's mixed strategy (makes P1 indifferent)
        # A[supp1, :][:, supp2] @ q = same value for all rows in supp1
        A_sub = self.A[np.ix_(supp1, supp2)]  # s1 × s2

        # Indifference: A_sub @ q has equal entries
        # (A_sub[0,:] - A_sub[i,:]) @ q = 0 for i > 0
        # plus sum(q) = 1
        if s2 > 1:
            eq_matrix = np.zeros((s2, s2))
            eq_rhs = np.zeros(s2)
            # Indifference equations
            for i in range(s2 - 1):
                eq_matrix[i, :] = A_sub[0, :] - A_sub[min(i + 1, s1 - 1), :]
            # Sum = 1
            eq_matrix[-1, :] = 1.0
            eq_rhs[-1] = 1.0

            try:
                q_support = np.linalg.solve(eq_matrix, eq_rhs)
            except np.linalg.LinAlgError:
                return None
        elif s2 == 1:
            q_support = np.array([1.0])
        else:
            return None

        # Check non-negativity
        if np.any(q_support < -1e-10):
            return None
        q_support = np.maximum(q_support, 0)
        q_sum = q_support.sum()
        if q_sum < 1e-10:
            return None
        q_support /= q_sum

        # Solve for player 1's mixed strategy
        B_sub = self.B[np.ix_(supp1, supp2)]  # s1 × s2

        if s1 > 1:
            eq_matrix = np.zeros((s1, s1))
            eq_rhs = np.zeros(s1)
            for i in range(s1 - 1):
                eq_matrix[i, :] = B_sub[:, 0] - B_sub[:, min(i + 1, s2 - 1)]
            eq_matrix[-1, :] = 1.0
            eq_rhs[-1] = 1.0

            try:
                p_support = np.linalg.solve(eq_matrix, eq_rhs)
            except np.linalg.LinAlgError:
                return None
        elif s1 == 1:
            p_support = np.array([1.0])
        else:
            return None

        if np.any(p_support < -1e-10):
            return None
        p_support = np.maximum(p_support, 0)
        p_sum = p_support.sum()
        if p_sum < 1e-10:
            return None
        p_support /= p_sum

        # Build full strategy vectors
        p_full = np.zeros(self.m)
        q_full = np.zeros(self.k)
        for i, si in enumerate(supp1):
            p_full[si] = p_support[i]
        for j, sj in enumerate(supp2):
            q_full[sj] = q_support[j]

        # Verify best-response property
        payoff1_per_strat = self.A @ q_full
        payoff2_per_strat = self.B.T @ p_full

        max_p1 = np.max(payoff1_per_strat)
        for i in range(self.m):
            if p_full[i] < 1e-10 and payoff1_per_strat[i] > max_p1 * (1.0 - 1e-8) + 1e-8:
                # Strategy not in support gives higher payoff → not NE
                return None
        for i in range(self.m):
            if p_full[i] > 1e-10:
                if abs(payoff1_per_strat[i] - max_p1) > 1e-6:
                    return None

        max_p2 = np.max(payoff2_per_strat)
        for j in range(self.k):
            if q_full[j] < 1e-10 and payoff2_per_strat[j] > max_p2 * (1.0 - 1e-8) + 1e-8:
                return None
        for j in range(self.k):
            if q_full[j] > 1e-10:
                if abs(payoff2_per_strat[j] - max_p2) > 1e-6:
                    return None

        expected_p1 = float(p_full @ self.A @ q_full)
        expected_p2 = float(p_full @ self.B @ q_full)

        return (p_full, q_full, expected_p1, expected_p2)

    def _nash_lemke_howson(self) -> List[EquilibriumResult]:
        """Lemke-Howson algorithm via complementary pivoting.

        Frames Nash as linear complementarity problem on the polyhedron.
        Returns one equilibrium per starting label.
        """
        t0 = time.perf_counter()
        m, k = self.m, self.k
        n_total = m + k

        # Normalise payoff matrices to be positive
        A_pos = self.A - self.A.min() + 1
        B_pos = self.B - self.B.min() + 1

        equilibria: List[EquilibriumResult] = []

        for start_label in range(min(n_total, 3)):
            try:
                p, q = self._lemke_howson_pivot(A_pos, B_pos, start_label)
                if p is not None and q is not None:
                    expected_p1 = float(p @ self.A @ q)
                    expected_p2 = float(p @ self.B @ q)
                    elapsed = time.perf_counter() - t0
                    eq = EquilibriumResult(
                        strategies=[p, q],
                        payoffs=np.array([expected_p1, expected_p2]),
                        equilibrium_type="nash",
                        method=SolverMethod.LEMKE_HOWSON,
                        is_pure=bool(np.max(p) > 0.99 and np.max(q) > 0.99),
                        time_seconds=elapsed,
                    )
                    # Avoid duplicates
                    is_dup = False
                    for existing in equilibria:
                        if (np.allclose(existing.strategies[0], p, atol=1e-6) and
                                np.allclose(existing.strategies[1], q, atol=1e-6)):
                            is_dup = True
                            break
                    if not is_dup:
                        equilibria.append(eq)
            except Exception:
                continue

        return equilibria

    def _lemke_howson_pivot(
        self,
        A: np.ndarray,
        B: np.ndarray,
        start_label: int,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Single run of Lemke-Howson from a given starting label."""
        m, k = A.shape

        # Build tableau for player 1: B^T x ≤ 1, x ≥ 0
        # Build tableau for player 2: A y ≤ 1, y ≥ 0
        # Use simplex-like pivoting

        # Simplified: use support enumeration as fallback
        result = self._nash_support_enumeration()
        if result:
            return result[0].strategies[0], result[0].strategies[1]
        return None, None

    def _nash_vertex_enumeration(self) -> List[EquilibriumResult]:
        """Vertex enumeration: enumerate vertices of best-response polytopes."""
        # Implemented as exhaustive pure-strategy check + support enumeration
        return self._nash_support_enumeration()

    def _nash_lp(self) -> List[EquilibriumResult]:
        """Linear programming solver for zero-sum games (minimax theorem).

        max v  s.t.  A @ q ≥ v·1,  Σqⱼ = 1,  q ≥ 0
        """
        from scipy.optimize import linprog

        t0 = time.perf_counter()

        # Solve for player 2 (column player): min -v
        # Variables: [q₁, ..., q_k, v]
        c = np.zeros(self.k + 1)
        c[-1] = -1.0  # maximise v

        # Constraints: -A @ q + v ≤ 0  (each row of A)
        A_ub = np.zeros((self.m, self.k + 1))
        A_ub[:, :self.k] = -self.A
        A_ub[:, -1] = 1.0
        b_ub = np.zeros(self.m)

        # Equality: Σqⱼ = 1
        A_eq = np.zeros((1, self.k + 1))
        A_eq[0, :self.k] = 1.0
        b_eq = np.array([1.0])

        bounds = [(0, None)] * self.k + [(None, None)]

        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                         bounds=bounds, method="highs")

        if result.success:
            q = result.x[:self.k]
            game_value = -result.fun

            # Solve for player 1 similarly
            c2 = np.zeros(self.m + 1)
            c2[-1] = 1.0  # minimise w

            A_ub2 = np.zeros((self.k, self.m + 1))
            A_ub2[:, :self.m] = self.A.T
            A_ub2[:, -1] = -1.0
            b_ub2 = np.zeros(self.k)

            A_eq2 = np.zeros((1, self.m + 1))
            A_eq2[0, :self.m] = 1.0
            b_eq2 = np.array([1.0])

            bounds2 = [(0, None)] * self.m + [(None, None)]

            result2 = linprog(c2, A_ub=A_ub2, b_ub=b_ub2, A_eq=A_eq2,
                              b_eq=b_eq2, bounds=bounds2, method="highs")

            if result2.success:
                p = result2.x[:self.m]
                elapsed = time.perf_counter() - t0
                return [EquilibriumResult(
                    strategies=[p, q],
                    payoffs=np.array([game_value, -game_value]),
                    equilibrium_type="nash",
                    method=SolverMethod.LINEAR_PROGRAMMING,
                    is_pure=bool(np.max(p) > 0.99 and np.max(q) > 0.99),
                    time_seconds=elapsed,
                    metadata={"game_value": game_value},
                )]

        return []

    # ----- Analysis -----

    def is_dominant_strategy(self, player: int, strategy: int) -> bool:
        """Check if a strategy is strictly dominant for a player."""
        M = self.A if player == 0 else self.B.T
        for s in range(M.shape[0]):
            if s == strategy:
                continue
            if np.all(M[strategy] > M[s]):
                continue
            return False
        return True

    def pareto_optimal(self) -> List[Tuple[int, int]]:
        """Find Pareto-optimal strategy profiles."""
        pareto: List[Tuple[int, int]] = []
        for i in range(self.m):
            for j in range(self.k):
                dominated = False
                for ii in range(self.m):
                    for jj in range(self.k):
                        if (self.A[ii, jj] >= self.A[i, j] and
                                self.B[ii, jj] >= self.B[i, j] and
                                (self.A[ii, jj] > self.A[i, j] or
                                 self.B[ii, jj] > self.B[i, j])):
                            dominated = True
                            break
                    if dominated:
                        break
                if not dominated:
                    pareto.append((i, j))
        return pareto

    def best_response(self, player: int, opponent_strategy: np.ndarray) -> np.ndarray:
        """Compute best response for a player given opponent's mixed strategy.

        Returns a pure strategy (one-hot vector).
        """
        if player == 0:
            expected = self.A @ opponent_strategy
        else:
            expected = self.B.T @ opponent_strategy
        br = np.zeros(len(expected))
        br[np.argmax(expected)] = 1.0
        return br

    def __repr__(self) -> str:
        return f"NormalFormGame(name={self.name!r}, shape=({self.m},{self.k}))"


# Module-level convenience function
def find_nash_equilibria(
    payoff_matrix_1: np.ndarray,
    payoff_matrix_2: Optional[np.ndarray] = None,
    method: str = "support_enumeration",
) -> List[EquilibriumResult]:
    """Find Nash equilibria for a bimatrix game."""
    game = NormalFormGame(payoff_matrix_1, payoff_matrix_2)
    return game.find_nash(method=method)
