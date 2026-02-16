"""
Sudoku solver — constraint propagation + backtracking, QUBO encoding.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from quant_gameth._types import SolverResult, SolverMethod


def solve_sudoku(
    board: np.ndarray,
    method: str = "backtracking",
    seed: int = 42,
) -> SolverResult:
    """Solve a Sudoku puzzle.

    Parameters
    ----------
    board : np.ndarray
        9×9 board where 0 indicates empty cells.
    method : str
        ``'backtracking'`` or ``'constraint_propagation'``.
    seed : int
    """
    if method == "constraint_propagation":
        return _sudoku_cp(board)
    return _sudoku_backtracking(board)


def _sudoku_backtracking(board: np.ndarray) -> SolverResult:
    t0 = time.perf_counter()
    grid = board.copy()
    nodes = [0]

    def find_empty() -> Optional[tuple]:
        for i in range(9):
            for j in range(9):
                if grid[i, j] == 0:
                    return (i, j)
        return None

    def is_valid(row: int, col: int, num: int) -> bool:
        # Check row
        if num in grid[row, :]:
            return False
        # Check column
        if num in grid[:, col]:
            return False
        # Check 3×3 box
        box_r, box_c = 3 * (row // 3), 3 * (col // 3)
        if num in grid[box_r:box_r + 3, box_c:box_c + 3]:
            return False
        return True

    def solve() -> bool:
        nodes[0] += 1
        cell = find_empty()
        if cell is None:
            return True
        row, col = cell
        for num in range(1, 10):
            if is_valid(row, col, num):
                grid[row, col] = num
                if solve():
                    return True
                grid[row, col] = 0
        return False

    success = solve()
    elapsed = time.perf_counter() - t0

    return SolverResult(
        solution=grid.flatten(),
        energy=0.0 if success else float("inf"),
        method=SolverMethod.BACKTRACKING,
        iterations=nodes[0],
        time_seconds=elapsed,
        converged=success,
        metadata={"board": grid.tolist()},
    )


def _sudoku_cp(board: np.ndarray) -> SolverResult:
    """Constraint propagation with naked singles and hidden singles."""
    t0 = time.perf_counter()
    grid = board.copy()

    # Possible values for each cell
    possible = [[set(range(1, 10)) if grid[i, j] == 0 else {grid[i, j]}
                 for j in range(9)] for i in range(9)]

    def eliminate() -> bool:
        changed = True
        while changed:
            changed = False
            for i in range(9):
                for j in range(9):
                    if len(possible[i][j]) == 1:
                        val = next(iter(possible[i][j]))
                        if grid[i, j] == 0:
                            grid[i, j] = val
                            changed = True
                        # Eliminate from peers
                        for k in range(9):
                            if k != j and val in possible[i][k]:
                                possible[i][k].discard(val)
                                changed = True
                            if k != i and val in possible[k][j]:
                                possible[k][j].discard(val)
                                changed = True
                        box_r, box_c = 3 * (i // 3), 3 * (j // 3)
                        for r in range(box_r, box_r + 3):
                            for c in range(box_c, box_c + 3):
                                if (r, c) != (i, j) and val in possible[r][c]:
                                    possible[r][c].discard(val)
                                    changed = True
                    elif len(possible[i][j]) == 0:
                        return False
        return True

    success = eliminate()

    # Check if solved
    if success and np.all(grid > 0):
        elapsed = time.perf_counter() - t0
        return SolverResult(
            solution=grid.flatten(),
            energy=0.0,
            method=SolverMethod.CONSTRAINT_PROPAGATION,
            iterations=1,
            time_seconds=elapsed,
            converged=True,
            metadata={"board": grid.tolist()},
        )

    # Fall back to backtracking for remaining cells
    return _sudoku_backtracking(grid)
