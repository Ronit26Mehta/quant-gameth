"""
Puzzle generators — Sudoku, N-Queens boards.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def generate_sudoku(
    difficulty: str = "medium",
    seed: int = 42,
) -> np.ndarray:
    """Generate a valid Sudoku puzzle.

    Parameters
    ----------
    difficulty : str
        ``'easy'`` (~35 clues), ``'medium'`` (~28), ``'hard'`` (~22).
    seed : int

    Returns
    -------
    np.ndarray
        9×9 board with 0 for empty cells.
    """
    rng = np.random.default_rng(seed)
    board = _generate_full_board(rng)

    clues = {"easy": 35, "medium": 28, "hard": 22}.get(difficulty, 28)
    n_remove = 81 - clues

    cells = list(range(81))
    rng.shuffle(cells)
    removed = 0

    for cell in cells:
        if removed >= n_remove:
            break
        r, c = divmod(cell, 9)
        old_val = board[r, c]
        board[r, c] = 0
        removed += 1

    return board


def _generate_full_board(rng: np.random.Generator) -> np.ndarray:
    """Generate a complete valid Sudoku board."""
    board = np.zeros((9, 9), dtype=int)

    def is_valid(row: int, col: int, num: int) -> bool:
        if num in board[row, :]:
            return False
        if num in board[:, col]:
            return False
        box_r, box_c = 3 * (row // 3), 3 * (col // 3)
        if num in board[box_r:box_r + 3, box_c:box_c + 3]:
            return False
        return True

    def fill(pos: int) -> bool:
        if pos == 81:
            return True
        r, c = divmod(pos, 9)
        nums = rng.permutation(9) + 1
        for n in nums:
            if is_valid(r, c, int(n)):
                board[r, c] = int(n)
                if fill(pos + 1):
                    return True
                board[r, c] = 0
        return False

    fill(0)
    return board


def generate_nqueens(n: int = 8) -> int:
    """Return board size for N-Queens (trivially n, for API consistency)."""
    return n
