"""
QUBO builder and Ising model conversion.

From Mathematical Foundations §D:
    f(x) = x^T Q x    where x ∈ {0,1}^n
    
    Ising conversion (§D.3):
    xᵢ = (σᵢ + 1)/2    where σᵢ ∈ {-1,+1}
    Jᵢⱼ = Qᵢⱼ/4,   hᵢ = (ΣⱼQᵢⱼ)/4
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from quant_gameth._types import EncodedProblem


class QUBOBuilder:
    """Incrementally build a QUBO problem.

    Usage::

        q = QUBOBuilder(4)
        q.add_linear(0, -2.0)        # diagonal term
        q.add_quadratic(0, 1, 1.5)   # off-diagonal
        q.add_one_hot([0,1,2], P=5)  # constraint
        problem = q.build()
    """

    def __init__(self, n_variables: int):
        self.n = n_variables
        self._Q = np.zeros((n_variables, n_variables), dtype=float)
        self._offset: float = 0.0
        self._labels: List[str] = [f"x{i}" for i in range(n_variables)]
        self._constraints_count: int = 0

    def add_linear(self, i: int, weight: float) -> "QUBOBuilder":
        """Add linear term wᵢ xᵢ (goes on diagonal Q[i,i])."""
        self._Q[i, i] += weight
        return self

    def add_quadratic(self, i: int, j: int, weight: float) -> "QUBOBuilder":
        """Add quadratic term wᵢⱼ xᵢxⱼ."""
        if i == j:
            self._Q[i, i] += weight
        else:
            # Store in upper triangle
            lo, hi = min(i, j), max(i, j)
            self._Q[lo, hi] += weight
        return self

    def add_one_hot(
        self, variables: List[int], penalty: float = 10.0
    ) -> "QUBOBuilder":
        """Constraint: exactly one of the variables is 1 (§D.2).

        Penalty: P(Σxᵢ - 1)² = P[Σᵢxᵢ - 2Σᵢ<ⱼ xᵢxⱼ + Σᵢxᵢ - 2Σᵢxᵢ + 1]
        """
        for i in variables:
            self._Q[i, i] += -penalty  # linear: -2P summed, but xᵢ²=xᵢ → shift
        for idx_a, va in enumerate(variables):
            for vb in variables[idx_a + 1:]:
                lo, hi = min(va, vb), max(va, vb)
                self._Q[lo, hi] += 2 * penalty
        self._offset += penalty
        self._constraints_count += 1
        return self

    def add_at_most_one(
        self, variables: List[int], penalty: float = 10.0
    ) -> "QUBOBuilder":
        """Constraint: at most one variable is 1."""
        for idx_a, va in enumerate(variables):
            for vb in variables[idx_a + 1:]:
                lo, hi = min(va, vb), max(va, vb)
                self._Q[lo, hi] += penalty
        self._constraints_count += 1
        return self

    def add_equality(
        self,
        variables: List[int],
        coefficients: List[float],
        target: float,
        penalty: float = 10.0,
    ) -> "QUBOBuilder":
        """Constraint: Σ cᵢxᵢ = target.

        Penalty: P(Σ cᵢxᵢ - target)²
        """
        n = len(variables)
        for i in range(n):
            self._Q[variables[i], variables[i]] += penalty * (
                coefficients[i] ** 2 - 2 * coefficients[i] * target
            )
        for i in range(n):
            for j in range(i + 1, n):
                lo = min(variables[i], variables[j])
                hi = max(variables[i], variables[j])
                self._Q[lo, hi] += 2 * penalty * coefficients[i] * coefficients[j]
        self._offset += penalty * target ** 2
        self._constraints_count += 1
        return self

    def add_inequality_leq(
        self,
        variables: List[int],
        coefficients: List[float],
        bound: float,
        n_slack_bits: int = 3,
        penalty: float = 10.0,
    ) -> "QUBOBuilder":
        """Constraint: Σ cᵢxᵢ ≤ bound (§D.2).

        Introduces binary slack variables: s = bound - Σcᵢxᵢ = Σⱼ 2ʲsⱼ
        """
        # Extend Q matrix for slack variables
        old_n = self.n
        new_n = old_n + n_slack_bits
        new_Q = np.zeros((new_n, new_n), dtype=float)
        new_Q[:old_n, :old_n] = self._Q
        self._Q = new_Q
        self.n = new_n

        # Slack variable labels
        for j in range(n_slack_bits):
            self._labels.append(f"slack_{self._constraints_count}_{j}")

        # Equality: Σcᵢxᵢ + Σ 2ʲsⱼ = bound
        all_vars = list(variables) + list(range(old_n, new_n))
        all_coeffs = list(coefficients) + [2 ** j for j in range(n_slack_bits)]
        self.add_equality(all_vars, all_coeffs, bound, penalty)
        return self

    def set_labels(self, labels: List[str]) -> "QUBOBuilder":
        """Set variable labels."""
        self._labels = labels
        return self

    def build(self, problem_type: str = "generic") -> EncodedProblem:
        """Build the final EncodedProblem."""
        h, J, offset = self.to_ising()
        return EncodedProblem(
            qubo_matrix=self._Q.copy(),
            ising_h=h,
            ising_J=J,
            n_variables=self.n,
            n_logical_qubits=self.n,
            offset=self._offset + offset,
            variable_labels=self._labels[:self.n],
            constraints_count=self._constraints_count,
            problem_type=problem_type,
        )

    def to_ising(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Convert QUBO to Ising model (§D.3).

        xᵢ = (σᵢ+1)/2 → Jᵢⱼ = Qᵢⱼ/4,  hᵢ = row_sum/4 + Qᵢᵢ/2
        """
        n = self.n
        # Make Q symmetric for conversion
        Q_sym = self._Q + np.triu(self._Q, 1).T

        J = np.zeros((n, n), dtype=float)
        h = np.zeros(n, dtype=float)
        offset = 0.0

        for i in range(n):
            for j in range(i + 1, n):
                J[i, j] = Q_sym[i, j] / 4.0
            h[i] = Q_sym[i, i] / 2.0
            for j in range(n):
                if i != j:
                    h[i] += Q_sym[i, j] / 4.0

        offset = np.sum(Q_sym) / 4.0 + np.trace(Q_sym) / 4.0

        return h, J, offset

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate QUBO objective f(x) = x^T Q x + offset."""
        return float(x @ self._Q @ x) + self._offset

    def to_cost_diagonal(self) -> np.ndarray:
        """Convert QUBO to a cost diagonal for QAOA.

        Returns array of length 2^n: cost_diagonal[x] = f(x).
        """
        dim = 1 << self.n
        diag = np.zeros(dim)
        for x_int in range(dim):
            x = np.array([(x_int >> i) & 1 for i in range(self.n)], dtype=float)
            diag[x_int] = self.evaluate(x)
        return diag

    @staticmethod
    def from_maxcut(adjacency: np.ndarray) -> "QUBOBuilder":
        """Create a MaxCut QUBO (§D.4).

        QUBO: min -Σ_{(i,j)∈E} (xᵢ + xⱼ - 2xᵢxⱼ)
        """
        n = len(adjacency)
        q = QUBOBuilder(n)
        for i in range(n):
            for j in range(i + 1, n):
                if adjacency[i, j] != 0:
                    w = adjacency[i, j]
                    q.add_linear(i, -w)
                    q.add_linear(j, -w)
                    q.add_quadratic(i, j, 2 * w)
        return q

    @staticmethod
    def from_graph_coloring(
        adjacency: np.ndarray, n_colors: int, penalty: float = 10.0
    ) -> "QUBOBuilder":
        """Create a graph coloring QUBO (§D.4).

        Variables: x_{i,c} = 1 if node i has color c.
        """
        n_nodes = len(adjacency)
        n_vars = n_nodes * n_colors
        q = QUBOBuilder(n_vars)
        labels = [f"node{i}_color{c}" for i in range(n_nodes) for c in range(n_colors)]
        q.set_labels(labels)

        # Each node exactly one color
        for i in range(n_nodes):
            node_vars = [i * n_colors + c for c in range(n_colors)]
            q.add_one_hot(node_vars, penalty)

        # Adjacent nodes different colors
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if adjacency[i, j] != 0:
                    for c in range(n_colors):
                        q.add_quadratic(
                            i * n_colors + c,
                            j * n_colors + c,
                            penalty,
                        )

        return q

    @staticmethod
    def from_tsp(
        distance_matrix: np.ndarray, penalty: float = 100.0
    ) -> "QUBOBuilder":
        """Create a TSP QUBO (§D.4).

        Variables: x_{i,t} = 1 if city i is visited at time t.
        """
        n = len(distance_matrix)
        n_vars = n * n
        q = QUBOBuilder(n_vars)
        labels = [f"city{i}_time{t}" for i in range(n) for t in range(n)]
        q.set_labels(labels)

        # Each city visited once
        for i in range(n):
            city_vars = [i * n + t for t in range(n)]
            q.add_one_hot(city_vars, penalty)

        # Each time slot one city
        for t in range(n):
            time_vars = [i * n + t for i in range(n)]
            q.add_one_hot(time_vars, penalty)

        # Distance cost: Σ d_{ij} x_{i,t} x_{j,t+1}
        for i in range(n):
            for j in range(n):
                if i != j and distance_matrix[i, j] > 0:
                    for t in range(n):
                        t_next = (t + 1) % n
                        q.add_quadratic(
                            i * n + t,
                            j * n + t_next,
                            distance_matrix[i, j],
                        )

        return q
