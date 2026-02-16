"""
MaxCut solver — QAOA, annealing, and exact brute-force.
"""

from __future__ import annotations

import time
from typing import Dict, Optional

import numpy as np

from quant_gameth._types import SolverResult, SolverMethod
from quant_gameth.encoders.qubo import QUBOBuilder


def solve_maxcut(
    adjacency: np.ndarray,
    method: str = "qaoa",
    qaoa_depth: int = 3,
    sa_steps: int = 5000,
    seed: int = 42,
) -> SolverResult:
    """Solve MaxCut problem.

    Parameters
    ----------
    adjacency : np.ndarray
        Adjacency/weight matrix, shape ``(n, n)``.
    method : str
        ``'qaoa'``, ``'annealing'``, ``'brute_force'``.
    qaoa_depth : int
    sa_steps : int
    seed : int
    """
    n = len(adjacency)

    if method == "brute_force" and n <= 20:
        return _maxcut_brute_force(adjacency)
    elif method == "annealing":
        return _maxcut_annealing(adjacency, sa_steps, seed)
    else:
        return _maxcut_qaoa(adjacency, qaoa_depth, seed)


def _evaluate_cut(adjacency: np.ndarray, partition: np.ndarray) -> float:
    """Evaluate the cut value for a given partition."""
    n = len(adjacency)
    cut = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if partition[i] != partition[j]:
                cut += adjacency[i, j]
    return cut


def _maxcut_brute_force(adjacency: np.ndarray) -> SolverResult:
    t0 = time.perf_counter()
    n = len(adjacency)
    best_cut = -np.inf
    best_partition = np.zeros(n, dtype=int)

    for x_int in range(1 << n):
        partition = np.array([(x_int >> i) & 1 for i in range(n)], dtype=int)
        cut = _evaluate_cut(adjacency, partition)
        if cut > best_cut:
            best_cut = cut
            best_partition = partition.copy()

    return SolverResult(
        solution=best_partition,
        energy=-best_cut,
        method=SolverMethod.BRUTE_FORCE,
        iterations=1 << n,
        time_seconds=time.perf_counter() - t0,
        converged=True,
        metadata={"cut_value": float(best_cut)},
    )


def _maxcut_qaoa(adjacency: np.ndarray, depth: int, seed: int) -> SolverResult:
    from quant_gameth.quantum.qaoa import QAOASolver

    n = len(adjacency)
    qubo = QUBOBuilder.from_maxcut(adjacency)
    cost_diag = qubo.to_cost_diagonal()

    solver = QAOASolver(n_qubits=n, depth=depth)
    result = solver.solve(cost_diag, seed=seed)

    # Extract partition
    best_state = int(np.argmin(cost_diag * np.abs(result.metadata.get("final_state", np.ones(1 << n))) ** 2
                                if "final_state" in result.metadata else cost_diag))
    partition = np.array([(best_state >> i) & 1 for i in range(n)], dtype=int)
    cut_value = _evaluate_cut(adjacency, partition)

    result.metadata["cut_value"] = float(cut_value)
    result.solution = partition
    return result


def _maxcut_annealing(adjacency: np.ndarray, n_steps: int, seed: int) -> SolverResult:
    from quant_gameth.quantum.annealing import simulated_annealing

    n = len(adjacency)
    qubo = QUBOBuilder.from_maxcut(adjacency)

    def energy_fn(x: np.ndarray) -> float:
        return qubo.evaluate(x)

    result = simulated_annealing(
        n_variables=n,
        energy_fn=energy_fn,
        n_steps=n_steps,
        seed=seed,
    )

    cut_value = _evaluate_cut(adjacency, result.solution.astype(int))
    result.metadata["cut_value"] = float(cut_value)
    return result
