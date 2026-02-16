"""
Graph coloring solver — QUBO + QAOA / classical backtracking.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from quant_gameth._types import SolverResult, SolverMethod
from quant_gameth.encoders.qubo import QUBOBuilder


def solve_graph_coloring(
    adjacency: np.ndarray,
    n_colors: int = 3,
    method: str = "backtracking",
    seed: int = 42,
) -> SolverResult:
    """Solve the graph coloring problem.

    Parameters
    ----------
    adjacency : np.ndarray
    n_colors : int
    method : str
        ``'backtracking'``, ``'qaoa'``, ``'annealing'``.
    seed : int
    """
    if method == "backtracking":
        return _coloring_backtracking(adjacency, n_colors)
    elif method == "qaoa":
        return _coloring_qaoa(adjacency, n_colors, seed)
    else:
        return _coloring_annealing(adjacency, n_colors, seed)


def _coloring_backtracking(adjacency: np.ndarray, n_colors: int) -> SolverResult:
    t0 = time.perf_counter()
    n = len(adjacency)
    colors = np.full(n, -1, dtype=int)
    nodes_tried = [0]

    def is_safe(node: int, color: int) -> bool:
        for j in range(n):
            if adjacency[node, j] and colors[j] == color:
                return False
        return True

    def backtrack(node: int) -> bool:
        nodes_tried[0] += 1
        if node == n:
            return True
        for c in range(n_colors):
            if is_safe(node, c):
                colors[node] = c
                if backtrack(node + 1):
                    return True
                colors[node] = -1
        return False

    success = backtrack(0)
    elapsed = time.perf_counter() - t0

    return SolverResult(
        solution=colors.copy(),
        energy=0.0 if success else float("inf"),
        method=SolverMethod.BACKTRACKING,
        iterations=nodes_tried[0],
        time_seconds=elapsed,
        converged=success,
        metadata={
            "n_colors": n_colors,
            "coloring": colors.tolist() if success else None,
        },
    )


def _coloring_qaoa(adjacency: np.ndarray, n_colors: int, seed: int) -> SolverResult:
    from quant_gameth.quantum.qaoa import QAOASolver

    qubo = QUBOBuilder.from_graph_coloring(adjacency, n_colors)
    problem = qubo.build("graph_coloring")
    n_qubits = problem.n_variables

    if n_qubits > 20:
        # Fallback to annealing for large problems
        return _coloring_annealing(adjacency, n_colors, seed)

    cost_diag = qubo.to_cost_diagonal()
    solver = QAOASolver(n_qubits=n_qubits, depth=2)
    result = solver.solve(cost_diag, seed=seed)
    return result


def _coloring_annealing(adjacency: np.ndarray, n_colors: int, seed: int) -> SolverResult:
    from quant_gameth.quantum.annealing import simulated_annealing

    qubo = QUBOBuilder.from_graph_coloring(adjacency, n_colors)

    def energy_fn(x: np.ndarray) -> float:
        return qubo.evaluate(x)

    result = simulated_annealing(
        n_variables=qubo.n,
        energy_fn=energy_fn,
        n_steps=5000,
        seed=seed,
    )
    return result
