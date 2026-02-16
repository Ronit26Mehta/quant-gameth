"""
demo_sudoku.py — Sudoku Solver Showcase
========================================
Run:  python -m quant_gameth.examples.demo_sudoku

Demonstrates:
  1. Puzzle generation (easy / medium / hard)
  2. Constraint propagation solver
  3. Backtracking solver
  4. Solution verification
"""

from __future__ import annotations

import numpy as np


def print_board(board: np.ndarray) -> None:
    """Pretty-print a 9×9 Sudoku board."""
    for i in range(9):
        if i % 3 == 0 and i != 0:
            print("  ------+-------+------")
        row = ""
        for j in range(9):
            if j % 3 == 0 and j != 0:
                row += " | "
            val = board[i, j]
            row += f" {val if val != 0 else '.'}"
        print(row)


def verify_solution(board: np.ndarray) -> bool:
    """Check if a Sudoku solution is valid."""
    for i in range(9):
        if len(set(board[i, :])) != 9 or set(board[i, :]) != set(range(1, 10)):
            return False
        if len(set(board[:, i])) != 9 or set(board[:, i]) != set(range(1, 10)):
            return False
    for br in range(3):
        for bc in range(3):
            block = board[br*3:(br+1)*3, bc*3:(bc+1)*3].flatten()
            if len(set(block)) != 9 or set(block) != set(range(1, 10)):
                return False
    return True


def main() -> None:
    print("=" * 70)
    print("  SUDOKU SOLVER DEMO — quant-gameth")
    print("=" * 70)

    from quant_gameth.generators.puzzles import generate_sudoku
    from quant_gameth.solvers.sudoku import solve_sudoku

    for difficulty in ["easy", "medium", "hard"]:
        print(f"\n▸ {difficulty.upper()} puzzle (seed=42)")

        board = generate_sudoku(difficulty=difficulty, seed=42)
        n_clues = np.count_nonzero(board)
        print(f"  Clues: {n_clues}/81")
        print_board(board)

        # Solve with constraint propagation
        result_cp = solve_sudoku(board, method="constraint_propagation")
        # Solve with backtracking
        result_bt = solve_sudoku(board, method="backtracking")

        print(f"\n  Constraint propagation: {'✓ Solved' if result_cp.converged else '✗ Failed'} "
              f"({result_cp.time_seconds*1000:.1f}ms, {result_cp.iterations} iterations)")
        print(f"  Backtracking          : {'✓ Solved' if result_bt.converged else '✗ Failed'} "
              f"({result_bt.time_seconds*1000:.1f}ms, {result_bt.iterations} iterations)")

        if result_bt.converged:
            solution = result_bt.solution.reshape(9, 9).astype(int)
            valid = verify_solution(solution)
            print(f"  Solution valid: {'✓' if valid else '✗'}")
            print("\n  Solution:")
            print_board(solution)

    print("\n" + "=" * 70)
    print("  ✓ Sudoku demo complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
