"""
Constraint-to-QUBO compiler — express high-level constraints as penalty terms.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from quant_gameth.encoders.qubo import QUBOBuilder
from quant_gameth._types import ConstraintType, EncodedProblem


class ConstraintCompiler:
    """Compile high-level constraints to QUBO penalty terms.

    Usage::

        cc = ConstraintCompiler(n_variables=9)
        cc.add_constraint(ConstraintType.EXACTLY_ONE, [0, 1, 2])
        cc.add_constraint(ConstraintType.INEQUALITY_LEQ, [3, 4], coefficients=[2, 3], bound=5)
        problem = cc.compile(default_penalty=10.0)
    """

    def __init__(self, n_variables: int):
        self.n_variables = n_variables
        self._constraints: List[Dict] = []
        self._objective_linear: Dict[int, float] = {}
        self._objective_quadratic: Dict[Tuple[int, int], float] = {}

    def set_objective(
        self,
        linear: Optional[Dict[int, float]] = None,
        quadratic: Optional[Dict[Tuple[int, int], float]] = None,
    ) -> "ConstraintCompiler":
        """Set the objective function to minimise."""
        if linear:
            self._objective_linear = linear
        if quadratic:
            self._objective_quadratic = quadratic
        return self

    def add_constraint(
        self,
        constraint_type: ConstraintType,
        variables: List[int],
        coefficients: Optional[List[float]] = None,
        bound: Optional[float] = None,
        penalty: Optional[float] = None,
        auxiliary_var: Optional[int] = None,
    ) -> "ConstraintCompiler":
        """Add a constraint.

        Parameters
        ----------
        constraint_type : ConstraintType
            Type of constraint.
        variables : list of int
            Indices of involved variables.
        coefficients : list of float or None
            Coefficients (for weighted constraints).
        bound : float or None
            Right-hand side for equality/inequality constraints.
        penalty : float or None
            Custom penalty weight (overrides default).
        auxiliary_var : int or None
            Index of auxiliary variable (for logical constraints).
        """
        self._constraints.append({
            "type": constraint_type,
            "variables": variables,
            "coefficients": coefficients,
            "bound": bound,
            "penalty": penalty,
            "auxiliary_var": auxiliary_var,
        })
        return self

    def compile(
        self,
        default_penalty: float = 10.0,
        problem_type: str = "compiled",
    ) -> EncodedProblem:
        """Compile all constraints into a QUBO.

        Returns
        -------
        EncodedProblem
            The compiled QUBO with Ising coefficients.
        """
        # Estimate total variables needed (including slack)
        slack_needed = 0
        for c in self._constraints:
            if c["type"] in (ConstraintType.INEQUALITY_LEQ, ConstraintType.INEQUALITY_GEQ):
                slack_needed += 4  # default 4 slack bits

        total_vars = self.n_variables + slack_needed
        builder = QUBOBuilder(total_vars)

        # Add objective
        for i, w in self._objective_linear.items():
            builder.add_linear(i, w)
        for (i, j), w in self._objective_quadratic.items():
            builder.add_quadratic(i, j, w)

        # Process constraints
        slack_idx = self.n_variables
        for c in self._constraints:
            p = c["penalty"] if c["penalty"] is not None else default_penalty
            ct = c["type"]
            variables = c["variables"]
            coefficients = c["coefficients"] or [1.0] * len(variables)

            if ct == ConstraintType.EXACTLY_ONE:
                builder.add_one_hot(variables, p)

            elif ct == ConstraintType.AT_MOST_ONE:
                builder.add_at_most_one(variables, p)

            elif ct == ConstraintType.ALL_DIFFERENT:
                # Pairwise different (via QUBO penalty for same value)
                for i in range(len(variables)):
                    for j in range(i + 1, len(variables)):
                        builder.add_quadratic(variables[i], variables[j], p)

            elif ct == ConstraintType.EQUALITY:
                bound = c["bound"] if c["bound"] is not None else 0.0
                builder.add_equality(variables, coefficients, bound, p)

            elif ct == ConstraintType.INEQUALITY_LEQ:
                bound = c["bound"] if c["bound"] is not None else 0.0
                n_slack = 4
                builder = self._rebuild_with_slack(builder, n_slack)
                slack_vars = list(range(slack_idx, slack_idx + n_slack))
                slack_coeffs = [2 ** j for j in range(n_slack)]
                all_vars = list(variables) + slack_vars
                all_coeffs = list(coefficients) + slack_coeffs
                builder.add_equality(all_vars, all_coeffs, bound, p)
                slack_idx += n_slack

            elif ct == ConstraintType.INEQUALITY_GEQ:
                # Σ cᵢxᵢ ≥ b  →  -Σ cᵢxᵢ ≤ -b
                bound = c["bound"] if c["bound"] is not None else 0.0
                neg_coefficients = [-ci for ci in coefficients]
                n_slack = 4
                builder = self._rebuild_with_slack(builder, n_slack)
                slack_vars = list(range(slack_idx, slack_idx + n_slack))
                slack_coeffs = [2 ** j for j in range(n_slack)]
                all_vars = list(variables) + slack_vars
                all_coeffs = neg_coefficients + slack_coeffs
                builder.add_equality(all_vars, all_coeffs, -bound, p)
                slack_idx += n_slack

            elif ct == ConstraintType.LOGICAL_AND:
                # z = x AND y  (§D.2)
                if c["auxiliary_var"] is not None:
                    z = c["auxiliary_var"]
                    x, y = variables[0], variables[1]
                    builder.add_linear(z, p)
                    builder.add_quadratic(x, y, p)
                    builder.add_quadratic(x, z, -p)
                    builder.add_quadratic(y, z, -p)

            elif ct == ConstraintType.LOGICAL_OR:
                # z = x OR y (§D.2)
                if c["auxiliary_var"] is not None:
                    z = c["auxiliary_var"]
                    x, y = variables[0], variables[1]
                    builder.add_linear(x, -p)
                    builder.add_linear(y, -p)
                    builder.add_linear(z, p)
                    builder.add_quadratic(x, y, p)
                    builder.add_quadratic(x, z, p)
                    builder.add_quadratic(y, z, p)
                    builder._offset += p

        return builder.build(problem_type)

    def _rebuild_with_slack(
        self, builder: QUBOBuilder, additional: int
    ) -> QUBOBuilder:
        """Ensure the builder has enough variables for slack."""
        needed = builder.n + additional
        if builder._Q.shape[0] < needed:
            old_n = builder.n
            new_Q = np.zeros((needed, needed), dtype=float)
            new_Q[:old_n, :old_n] = builder._Q[:old_n, :old_n]
            builder._Q = new_Q
            builder.n = needed
            while len(builder._labels) < needed:
                builder._labels.append(f"slack_{len(builder._labels)}")
        return builder
