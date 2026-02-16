"""
N-Queens solver — backtracking, simulated annealing.
"""

from __future__ import annotations

import time
from typing import List

import numpy as np

from quant_gameth._types import SolverResult, SolverMethod


def solve_nqueens(
    n: int = 8,
    method: str = "backtracking",
    find_all: bool = False,
    seed: int = 42,
) -> SolverResult:
    """Solve the N-Queens problem.

    Parameters
    ----------
    n : int
        Board size.
    method : str
        ``'backtracking'`` or ``'annealing'``.
    find_all : bool
        If True, find all solutions (backtracking only).
    seed : int
    """
    if method == "backtracking":
        return _nqueens_backtracking(n, find_all)
    return _nqueens_annealing(n, seed)


def _nqueens_backtracking(n: int, find_all: bool) -> SolverResult:
    t0 = time.perf_counter()
    solutions: List[np.ndarray] = []
    queens = np.full(n, -1, dtype=int)  # queens[row] = column
    nodes = [0]

    def is_safe(row: int, col: int) -> bool:
        for r in range(row):
            c = queens[r]
            if c == col or abs(c - col) == abs(r - row):
                return False
        return True

    def solve(row: int) -> bool:
        nodes[0] += 1
        if row == n:
            solutions.append(queens.copy())
            return not find_all  # stop if not finding all
        for col in range(n):
            if is_safe(row, col):
                queens[row] = col
                if solve(row + 1):
                    return True
                queens[row] = -1
        return False

    solve(0)
    elapsed = time.perf_counter() - t0

    first = solutions[0] if solutions else np.full(n, -1, dtype=int)

    return SolverResult(
        solution=first,
        energy=0.0 if solutions else float("inf"),
        method=SolverMethod.BACKTRACKING,
        iterations=nodes[0],
        time_seconds=elapsed,
        converged=bool(solutions),
        metadata={
            "n_solutions": len(solutions),
            "all_solutions": [s.tolist() for s in solutions] if find_all else None,
        },
    )


def _nqueens_annealing(n: int, seed: int) -> SolverResult:
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)

    queens = rng.permutation(n)  # one queen per row, queens[row] = col

    def conflicts(q: np.ndarray) -> int:
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                if abs(q[i] - q[j]) == abs(i - j):
                    count += 1
        return count

    current_conflicts = conflicts(queens)
    best_queens = queens.copy()
    best_conflicts = current_conflicts

    for step in range(10 * n * n):
        if current_conflicts == 0:
            break
        T = max(0.01, 2.0 * (1.0 - step / (10 * n * n)))

        # Swap two random columns
        i, j = rng.choice(n, 2, replace=False)
        new_queens = queens.copy()
        new_queens[i], new_queens[j] = new_queens[j], new_queens[i]
        new_conflicts = conflicts(new_queens)

        delta = new_conflicts - current_conflicts
        if delta <= 0 or rng.random() < np.exp(-delta / T):
            queens = new_queens
            current_conflicts = new_conflicts
            if current_conflicts < best_conflicts:
                best_conflicts = current_conflicts
                best_queens = queens.copy()

    return SolverResult(
        solution=best_queens,
        energy=float(best_conflicts),
        method=SolverMethod.SIMULATED_ANNEALING,
        iterations=10 * n * n,
        time_seconds=time.perf_counter() - t0,
        converged=best_conflicts == 0,
        metadata={"conflicts": int(best_conflicts)},
    )
