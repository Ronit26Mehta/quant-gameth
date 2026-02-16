"""
Knapsack solver — dynamic programming, QUBO, branch-and-bound.
"""

from __future__ import annotations

import time
from typing import List

import numpy as np

from quant_gameth._types import SolverResult, SolverMethod
from quant_gameth.encoders.qubo import QUBOBuilder


def solve_knapsack(
    values: np.ndarray,
    weights: np.ndarray,
    capacity: float,
    method: str = "dynamic_programming",
    seed: int = 42,
) -> SolverResult:
    """Solve 0-1 knapsack problem.

    Parameters
    ----------
    values : np.ndarray
        Value of each item.
    weights : np.ndarray
        Weight of each item.
    capacity : float
        Knapsack capacity.
    method : str
        ``'dynamic_programming'``, ``'branch_and_bound'``, ``'qaoa'``, ``'annealing'``.
    seed : int
    """
    if method == "dynamic_programming":
        return _knapsack_dp(values, weights, capacity)
    elif method == "branch_and_bound":
        return _knapsack_bb(values, weights, capacity)
    else:
        return _knapsack_quantum(values, weights, capacity, method, seed)


def _knapsack_dp(
    values: np.ndarray, weights: np.ndarray, capacity: float
) -> SolverResult:
    """Standard O(nW) dynamic programming."""
    t0 = time.perf_counter()
    n = len(values)
    W = int(capacity)

    dp = np.zeros((n + 1, W + 1))
    for i in range(1, n + 1):
        w_i = int(weights[i - 1])
        v_i = values[i - 1]
        for w in range(W + 1):
            dp[i, w] = dp[i - 1, w]
            if w_i <= w:
                dp[i, w] = max(dp[i, w], dp[i - 1, w - w_i] + v_i)

    # Backtrack to find solution
    solution = np.zeros(n, dtype=int)
    w = W
    for i in range(n, 0, -1):
        if dp[i, w] != dp[i - 1, w]:
            solution[i - 1] = 1
            w -= int(weights[i - 1])

    elapsed = time.perf_counter() - t0
    total_value = float(dp[n, W])

    return SolverResult(
        solution=solution,
        energy=-total_value,
        method=SolverMethod.DYNAMIC_PROGRAMMING,
        iterations=n * W,
        time_seconds=elapsed,
        converged=True,
        metadata={
            "total_value": total_value,
            "total_weight": float(np.sum(solution * weights)),
            "capacity": capacity,
        },
    )


def _knapsack_bb(
    values: np.ndarray, weights: np.ndarray, capacity: float
) -> SolverResult:
    """Branch and bound with LP relaxation upper bound."""
    t0 = time.perf_counter()
    n = len(values)

    # Sort by value/weight ratio
    ratios = values / np.maximum(weights, 1e-14)
    order = np.argsort(-ratios)

    best_value = [0.0]
    best_solution = [np.zeros(n, dtype=int)]
    nodes_explored = [0]

    def upper_bound(idx: int, current_weight: float, current_value: float) -> float:
        """Fractional knapsack upper bound."""
        ub = current_value
        remaining = capacity - current_weight
        for i in range(idx, n):
            item = order[i]
            if weights[item] <= remaining:
                ub += values[item]
                remaining -= weights[item]
            else:
                ub += values[item] * remaining / max(weights[item], 1e-14)
                break
        return ub

    def branch(idx: int, current_weight: float, current_value: float,
               current_sol: np.ndarray) -> None:
        nodes_explored[0] += 1
        if current_value > best_value[0]:
            best_value[0] = current_value
            best_solution[0] = current_sol.copy()

        if idx >= n:
            return

        # Prune
        if upper_bound(idx, current_weight, current_value) <= best_value[0]:
            return

        item = order[idx]
        # Include
        if current_weight + weights[item] <= capacity:
            current_sol[item] = 1
            branch(idx + 1, current_weight + weights[item],
                   current_value + values[item], current_sol)
            current_sol[item] = 0

        # Exclude
        branch(idx + 1, current_weight, current_value, current_sol)

    branch(0, 0.0, 0.0, np.zeros(n, dtype=int))
    elapsed = time.perf_counter() - t0

    return SolverResult(
        solution=best_solution[0],
        energy=-best_value[0],
        method=SolverMethod.BRANCH_AND_BOUND,
        iterations=nodes_explored[0],
        time_seconds=elapsed,
        converged=True,
        metadata={
            "total_value": best_value[0],
            "total_weight": float(np.sum(best_solution[0] * weights)),
        },
    )


def _knapsack_quantum(
    values: np.ndarray, weights: np.ndarray, capacity: float,
    method: str, seed: int,
) -> SolverResult:
    from quant_gameth.quantum.annealing import simulated_annealing

    n = len(values)
    penalty = float(np.max(values)) * 10

    builder = QUBOBuilder(n)
    for i in range(n):
        builder.add_linear(i, -values[i])

    # Capacity constraint via penalty
    n_slack = max(1, int(np.ceil(np.log2(max(capacity, 2)))))
    builder.add_inequality_leq(
        list(range(n)), weights.tolist(), capacity,
        n_slack_bits=n_slack, penalty=penalty,
    )

    def energy_fn(x: np.ndarray) -> float:
        return builder.evaluate(x[:n])

    result = simulated_annealing(
        n_variables=builder.n,
        energy_fn=lambda x: builder.evaluate(x),
        n_steps=5000,
        seed=seed,
    )

    solution = np.round(result.solution[:n]).astype(int)
    total_value = float(np.sum(solution * values))
    total_weight = float(np.sum(solution * weights))

    if total_weight > capacity:
        # Repair: remove items greedily
        ratios = values / np.maximum(weights, 1e-14)
        for idx in np.argsort(ratios):
            if total_weight <= capacity:
                break
            if solution[idx] == 1:
                solution[idx] = 0
                total_weight -= weights[idx]
                total_value -= values[idx]

    result.solution = solution
    result.metadata["total_value"] = total_value
    result.metadata["total_weight"] = total_weight
    return result
