"""
Decomposition strategies — divide-and-conquer, sliding window, hierarchical.

These utilities break large combinatorial problems into sub-problems
that can be solved independently (or with boundary constraints) and
then stitched back together.  They are backend-agnostic and work with
any solver that accepts a sub-problem and returns a SolverResult.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from quant_gameth._types import SolverResult, SolverMethod


# ---------------------------------------------------------------------------
# Divide-and-conquer
# ---------------------------------------------------------------------------

def divide_and_conquer(
    problem: np.ndarray,
    solver_fn: Callable[[np.ndarray, int], SolverResult],
    partition_fn: Optional[Callable[[np.ndarray], List[np.ndarray]]] = None,
    max_sub_size: int = 12,
    seed: int = 42,
) -> SolverResult:
    """Recursively partition a problem, solve sub-problems, and merge.

    Parameters
    ----------
    problem : np.ndarray
        Adjacency or cost matrix representing the full problem.
    solver_fn : callable
        ``solver_fn(sub_problem, seed) -> SolverResult`` — solver for
        a sub‐problem of manageable size.
    partition_fn : callable or None
        ``partition_fn(problem) -> list[np.ndarray]`` — custom partitioner.
        If None, uses a balanced bisection based on spectral ordering.
    max_sub_size : int
        Maximum sub-problem size that should be solved directly.
    seed : int
    """
    t0 = time.perf_counter()
    n = len(problem)

    # Base case: small enough to solve directly
    if n <= max_sub_size:
        result = solver_fn(problem, seed)
        result.metadata["decomposition"] = "direct"
        return result

    # Partition
    if partition_fn is not None:
        partitions = partition_fn(problem)
    else:
        partitions = _spectral_bisection(problem)

    # Solve each partition
    sub_results: List[SolverResult] = []
    sub_mappings: List[np.ndarray] = []

    for idx, part in enumerate(partitions):
        sub_matrix = problem[np.ix_(part, part)]
        sub_result = divide_and_conquer(
            sub_matrix, solver_fn, partition_fn, max_sub_size,
            seed=seed + idx * 1000,
        )
        sub_results.append(sub_result)
        sub_mappings.append(part)

    # Merge solutions back into full problem space
    full_solution = np.zeros(n, dtype=int)
    total_energy = 0.0

    for sub_result, mapping in zip(sub_results, sub_mappings):
        for local_idx, global_idx in enumerate(mapping):
            if local_idx < len(sub_result.solution):
                full_solution[global_idx] = sub_result.solution[local_idx]
        total_energy += sub_result.energy

    # Account for cross-partition edges (boundary penalty)
    cross_energy = _compute_cross_energy(problem, full_solution, sub_mappings)
    total_energy += cross_energy

    elapsed = time.perf_counter() - t0

    return SolverResult(
        solution=full_solution,
        energy=total_energy,
        method=SolverMethod.LOCAL_SEARCH,
        iterations=sum(r.iterations for r in sub_results),
        time_seconds=elapsed,
        converged=all(r.converged for r in sub_results),
        metadata={
            "decomposition": "divide_and_conquer",
            "n_partitions": len(partitions),
            "partition_sizes": [len(p) for p in partitions],
            "cross_energy": cross_energy,
        },
    )


def _spectral_bisection(problem: np.ndarray) -> List[np.ndarray]:
    """Partition via the Fiedler vector (second-smallest eigenvector of Laplacian)."""
    n = len(problem)
    degree = np.diag(problem.sum(axis=1))
    laplacian = degree - problem

    try:
        eigvals, eigvecs = np.linalg.eigh(laplacian)
        fiedler = eigvecs[:, 1]  # second smallest eigenvector
        median = np.median(fiedler)
        part_a = np.where(fiedler <= median)[0]
        part_b = np.where(fiedler > median)[0]
    except np.linalg.LinAlgError:
        # Fallback: simple halving
        mid = n // 2
        part_a = np.arange(mid)
        part_b = np.arange(mid, n)

    if len(part_a) == 0:
        part_a = np.array([0])
        part_b = np.arange(1, n)
    if len(part_b) == 0:
        part_b = np.array([n - 1])
        part_a = np.arange(n - 1)

    return [part_a, part_b]


def _compute_cross_energy(
    problem: np.ndarray,
    solution: np.ndarray,
    partitions: List[np.ndarray],
) -> float:
    """Compute energy contribution from edges crossing partition boundaries."""
    cross = 0.0
    n = len(problem)
    # Build partition membership
    membership = np.full(n, -1, dtype=int)
    for pid, part in enumerate(partitions):
        for idx in part:
            membership[idx] = pid

    for i in range(n):
        for j in range(i + 1, n):
            if membership[i] != membership[j] and problem[i, j] != 0:
                if solution[i] != solution[j]:
                    cross += problem[i, j]
    return cross


# ---------------------------------------------------------------------------
# Sliding window
# ---------------------------------------------------------------------------

def sliding_window(
    variables: np.ndarray,
    cost_fn: Callable[[np.ndarray], float],
    solver_fn: Callable[[np.ndarray, Callable, int], SolverResult],
    window_size: int = 10,
    stride: int = 5,
    n_sweeps: int = 3,
    seed: int = 42,
) -> SolverResult:
    """Optimise a large variable vector using overlapping local windows.

    At each position the solver optimises ``window_size`` variables
    while keeping the rest fixed.  Multiple sweeps refine the solution.

    Parameters
    ----------
    variables : np.ndarray
        Initial variable assignment (binary or continuous).
    cost_fn : callable
        ``cost_fn(full_variables) -> float``
    solver_fn : callable
        ``solver_fn(sub_variables, sub_cost_fn, seed) -> SolverResult``
    window_size : int
    stride : int
    n_sweeps : int
    seed : int
    """
    t0 = time.perf_counter()
    n = len(variables)
    current = variables.copy()
    best_cost = cost_fn(current)
    best_solution = current.copy()
    total_iterations = 0

    for sweep in range(n_sweeps):
        for start in range(0, n - window_size + 1, stride):
            end = start + window_size
            window_indices = list(range(start, end))

            # Sub-cost function: fix everything outside the window
            def sub_cost(sub_vars: np.ndarray, _indices=window_indices,
                         _current=current) -> float:
                temp = _current.copy()
                for i, idx in enumerate(_indices):
                    temp[idx] = sub_vars[i]
                return cost_fn(temp)

            sub_result = solver_fn(
                current[start:end], sub_cost, seed + sweep * 1000 + start,
            )

            # Accept if improved
            trial = current.copy()
            for i, idx in enumerate(window_indices):
                if i < len(sub_result.solution):
                    trial[idx] = sub_result.solution[i]

            trial_cost = cost_fn(trial)
            if trial_cost < best_cost:
                best_cost = trial_cost
                best_solution = trial.copy()
            current = trial
            total_iterations += sub_result.iterations

    elapsed = time.perf_counter() - t0

    return SolverResult(
        solution=best_solution,
        energy=best_cost,
        method=SolverMethod.LOCAL_SEARCH,
        iterations=total_iterations,
        time_seconds=elapsed,
        converged=True,
        metadata={
            "decomposition": "sliding_window",
            "window_size": window_size,
            "stride": stride,
            "n_sweeps": n_sweeps,
        },
    )


# ---------------------------------------------------------------------------
# Hierarchical coarsening
# ---------------------------------------------------------------------------

def hierarchical_solve(
    adjacency: np.ndarray,
    solver_fn: Callable[[np.ndarray, int], SolverResult],
    n_levels: int = 3,
    coarsen_ratio: float = 0.5,
    seed: int = 42,
) -> SolverResult:
    """Multi-level hierarchical solver: coarsen → solve → refine.

    1. **Coarsen** the graph by merging neighbouring nodes.
    2. **Solve** the coarsened problem (recursively or directly).
    3. **Uncoarsen** and refine the solution with local search.

    Parameters
    ----------
    adjacency : np.ndarray
    solver_fn : callable
        ``solver_fn(adj, seed) -> SolverResult``
    n_levels : int
    coarsen_ratio : float
        Target reduction per level (0.5 = halve the nodes).
    seed : int
    """
    t0 = time.perf_counter()

    hierarchy = _build_hierarchy(adjacency, n_levels, coarsen_ratio, seed)

    # Solve at the coarsest level
    coarsest_adj, _ = hierarchy[-1]
    result = solver_fn(coarsest_adj, seed)

    # Uncoarsen and refine at each level
    solution = result.solution
    total_iters = result.iterations

    for level in range(len(hierarchy) - 2, -1, -1):
        adj_level, mapping = hierarchy[level]
        n_level = len(adj_level)

        # Project solution from coarser level
        projected = np.zeros(n_level, dtype=int)
        for fine_idx, coarse_idx in enumerate(mapping):
            if coarse_idx < len(solution):
                projected[fine_idx] = solution[coarse_idx]

        # Local refinement: greedy 1-flip
        projected, flips = _local_refine(adj_level, projected)
        total_iters += flips
        solution = projected

    elapsed = time.perf_counter() - t0

    return SolverResult(
        solution=solution,
        energy=_evaluate_partition(adjacency, solution),
        method=SolverMethod.LOCAL_SEARCH,
        iterations=total_iters,
        time_seconds=elapsed,
        converged=True,
        metadata={
            "decomposition": "hierarchical",
            "n_levels": len(hierarchy),
            "level_sizes": [len(h[0]) for h in hierarchy],
        },
    )


def _build_hierarchy(
    adjacency: np.ndarray,
    n_levels: int,
    coarsen_ratio: float,
    seed: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Build the coarsening hierarchy.

    Returns list of (adjacency, mapping_to_coarser) tuples,
    from finest to coarsest.
    """
    rng = np.random.default_rng(seed)
    hierarchy: List[Tuple[np.ndarray, np.ndarray]] = []
    current = adjacency

    for level in range(n_levels):
        n = len(current)
        target = max(4, int(n * coarsen_ratio))
        if n <= target:
            identity_map = np.arange(n)
            hierarchy.append((current, identity_map))
            break

        # Heavy-edge matching
        matched = np.full(n, False)
        mapping = np.full(n, -1, dtype=int)
        coarse_idx = 0

        perm = rng.permutation(n)
        for i in perm:
            if matched[i]:
                continue
            # Find heaviest unmatched neighbour
            neighbours = np.where((current[i] > 0) & ~matched)[0]
            if len(neighbours) > 0:
                best = neighbours[np.argmax(current[i, neighbours])]
                mapping[i] = coarse_idx
                mapping[best] = coarse_idx
                matched[i] = True
                matched[best] = True
            else:
                mapping[i] = coarse_idx
                matched[i] = True
            coarse_idx += 1

        # Assign any remaining
        for i in range(n):
            if mapping[i] == -1:
                mapping[i] = coarse_idx
                coarse_idx += 1

        # Build coarser adjacency
        n_coarse = coarse_idx
        coarse_adj = np.zeros((n_coarse, n_coarse))
        for i in range(n):
            for j in range(i + 1, n):
                if current[i, j] != 0:
                    ci, cj = mapping[i], mapping[j]
                    if ci != cj:
                        coarse_adj[ci, cj] += current[i, j]
                        coarse_adj[cj, ci] += current[i, j]

        hierarchy.append((current, mapping))
        current = coarse_adj

    # Add coarsest level with identity mapping
    hierarchy.append((current, np.arange(len(current))))

    return hierarchy


def _local_refine(
    adjacency: np.ndarray,
    solution: np.ndarray,
    max_passes: int = 5,
) -> Tuple[np.ndarray, int]:
    """Greedy 1-flip local refinement for partition/coloring."""
    n = len(adjacency)
    current = solution.copy()
    best = current.copy()
    best_val = _evaluate_partition(adjacency, best)
    total_flips = 0

    for _ in range(max_passes):
        improved = False
        for i in range(n):
            old = current[i]
            current[i] = 1 - old  # flip
            val = _evaluate_partition(adjacency, current)
            if val < best_val:
                best_val = val
                best = current.copy()
                improved = True
                total_flips += 1
            else:
                current[i] = old
        if not improved:
            break

    return best, total_flips


def _evaluate_partition(
    adjacency: np.ndarray,
    partition: np.ndarray,
) -> float:
    """Evaluate a binary partition as negative cut value (for minimisation)."""
    n = len(adjacency)
    cut = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if partition[i] != partition[j]:
                cut += adjacency[i, j]
    return -cut  # negative so minimisation = max-cut
