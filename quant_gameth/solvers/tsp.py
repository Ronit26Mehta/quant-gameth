"""
TSP solver — QUBO/QAOA, simulated annealing, nearest neighbor, 2-opt.
"""

from __future__ import annotations

import time
from typing import List, Optional

import numpy as np

from quant_gameth._types import SolverResult, SolverMethod
from quant_gameth.encoders.qubo import QUBOBuilder


def solve_tsp(
    distance_matrix: np.ndarray,
    method: str = "two_opt",
    seed: int = 42,
) -> SolverResult:
    """Solve Traveling Salesperson Problem.

    Parameters
    ----------
    distance_matrix : np.ndarray
        Distance/cost matrix, shape ``(n, n)``.
    method : str
        ``'brute_force'``, ``'nearest_neighbor'``, ``'two_opt'``,
        ``'annealing'``, ``'qaoa'``.
    seed : int
    """
    n = len(distance_matrix)

    if method == "brute_force" and n <= 12:
        return _tsp_brute_force(distance_matrix)
    elif method == "nearest_neighbor":
        return _tsp_nearest_neighbor(distance_matrix, seed)
    elif method == "two_opt":
        return _tsp_two_opt(distance_matrix, seed)
    elif method == "qaoa" and n <= 8:
        return _tsp_qaoa(distance_matrix, seed)
    else:
        return _tsp_annealing(distance_matrix, seed)


def _tour_length(dist: np.ndarray, tour: np.ndarray) -> float:
    total = 0.0
    n = len(tour)
    for i in range(n):
        total += dist[tour[i], tour[(i + 1) % n]]
    return total


def _tsp_brute_force(dist: np.ndarray) -> SolverResult:
    from itertools import permutations

    t0 = time.perf_counter()
    n = len(dist)
    best_tour = None
    best_cost = float("inf")
    count = 0

    # Fix first city to reduce search space
    for perm in permutations(range(1, n)):
        tour = np.array([0] + list(perm), dtype=int)
        cost = _tour_length(dist, tour)
        if cost < best_cost:
            best_cost = cost
            best_tour = tour.copy()
        count += 1

    return SolverResult(
        solution=best_tour,
        energy=best_cost,
        method=SolverMethod.BRUTE_FORCE,
        iterations=count,
        time_seconds=time.perf_counter() - t0,
        converged=True,
        metadata={"tour_length": best_cost, "tour": best_tour.tolist()},
    )


def _tsp_nearest_neighbor(dist: np.ndarray, seed: int) -> SolverResult:
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    n = len(dist)

    best_tour = None
    best_cost = float("inf")

    for start in range(n):
        visited = np.zeros(n, dtype=bool)
        tour = [start]
        visited[start] = True

        for _ in range(n - 1):
            current = tour[-1]
            # Find nearest unvisited
            dists = dist[current].copy()
            dists[visited] = float("inf")
            next_city = int(np.argmin(dists))
            tour.append(next_city)
            visited[next_city] = True

        tour_arr = np.array(tour, dtype=int)
        cost = _tour_length(dist, tour_arr)
        if cost < best_cost:
            best_cost = cost
            best_tour = tour_arr.copy()

    return SolverResult(
        solution=best_tour,
        energy=best_cost,
        method=SolverMethod.GREEDY,
        iterations=n,
        time_seconds=time.perf_counter() - t0,
        converged=True,
        metadata={"tour_length": best_cost, "tour": best_tour.tolist()},
    )


def _tsp_two_opt(dist: np.ndarray, seed: int) -> SolverResult:
    """2-opt local search starting from nearest neighbor solution."""
    t0 = time.perf_counter()
    nn_result = _tsp_nearest_neighbor(dist, seed)
    tour = nn_result.solution.copy()
    n = len(tour)
    improved = True
    iterations = 0

    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                # Compute improvement from reversing segment [i:j+1]
                old_cost = (dist[tour[i - 1], tour[i]] +
                            dist[tour[j], tour[(j + 1) % n]])
                new_cost = (dist[tour[i - 1], tour[j]] +
                            dist[tour[i], tour[(j + 1) % n]])
                if new_cost < old_cost - 1e-10:
                    tour[i:j + 1] = tour[i:j + 1][::-1]
                    improved = True
                iterations += 1

    cost = _tour_length(dist, tour)
    return SolverResult(
        solution=tour,
        energy=cost,
        method=SolverMethod.LOCAL_SEARCH,
        iterations=iterations,
        time_seconds=time.perf_counter() - t0,
        converged=True,
        metadata={"tour_length": cost, "tour": tour.tolist()},
    )


def _tsp_annealing(dist: np.ndarray, seed: int) -> SolverResult:
    from quant_gameth.quantum.annealing import simulated_annealing

    n = len(dist)
    rng = np.random.default_rng(seed)

    # Use permutation-based SA
    t0 = time.perf_counter()
    tour = np.arange(n)
    rng.shuffle(tour)
    best_tour = tour.copy()
    best_cost = _tour_length(dist, tour)
    current_cost = best_cost

    for step in range(10000):
        T = max(0.01, 10.0 * (1.0 - step / 10000))
        # Random swap
        i, j = sorted(rng.choice(n, 2, replace=False))
        # 2-opt move
        new_tour = tour.copy()
        new_tour[i:j + 1] = new_tour[i:j + 1][::-1]
        new_cost = _tour_length(dist, new_tour)

        delta = new_cost - current_cost
        if delta < 0 or rng.random() < np.exp(-delta / T):
            tour = new_tour
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_tour = tour.copy()

    return SolverResult(
        solution=best_tour,
        energy=best_cost,
        method=SolverMethod.SIMULATED_ANNEALING,
        iterations=10000,
        time_seconds=time.perf_counter() - t0,
        converged=True,
        metadata={"tour_length": best_cost, "tour": best_tour.tolist()},
    )


def _tsp_qaoa(dist: np.ndarray, seed: int) -> SolverResult:
    qubo = QUBOBuilder.from_tsp(dist)

    from quant_gameth.quantum.annealing import simulated_annealing

    def energy_fn(x: np.ndarray) -> float:
        return qubo.evaluate(x)

    result = simulated_annealing(
        n_variables=qubo.n,
        energy_fn=energy_fn,
        n_steps=8000,
        seed=seed,
    )
    return result
