"""
Minimax solver with alpha-beta pruning, transposition table, and iterative deepening.

Standard zero-sum game solver for two-player adversarial games.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Tuple, Any

import numpy as np

from quant_gameth._types import SolverResult, SolverMethod


def minimax_solve(
    evaluate_fn: Callable[[Any], float],
    get_moves_fn: Callable[[Any], List[Any]],
    apply_move_fn: Callable[[Any, Any], Any],
    initial_state: Any,
    max_depth: int = 10,
    maximizing: bool = True,
    use_alpha_beta: bool = True,
    use_transposition_table: bool = True,
) -> SolverResult:
    """Minimax search with alpha-beta pruning.

    Parameters
    ----------
    evaluate_fn : callable
        ``evaluate_fn(state) -> float`` — heuristic evaluation function.
    get_moves_fn : callable
        ``get_moves_fn(state) -> List[move]`` — legal moves from state.
    apply_move_fn : callable
        ``apply_move_fn(state, move) -> new_state``.
    initial_state : Any
        Root game state.
    max_depth : int
        Maximum search depth.
    maximizing : bool
        Whether root player is maximising.
    use_alpha_beta : bool
        Enable alpha-beta pruning.
    use_transposition_table : bool
        Cache evaluated positions.

    Returns
    -------
    SolverResult
        ``.solution`` contains the best move index, ``.energy`` is the minimax value.
    """
    t0 = time.perf_counter()
    tt: Dict[int, Tuple[float, int]] = {} if use_transposition_table else {}
    nodes_searched = [0]

    def _minimax(
        state: Any,
        depth: int,
        alpha: float,
        beta: float,
        is_max: bool,
    ) -> float:
        nodes_searched[0] += 1

        # Transposition table lookup
        state_hash = hash(str(state))
        if use_transposition_table and state_hash in tt:
            cached_val, cached_depth = tt[state_hash]
            if cached_depth >= depth:
                return cached_val

        moves = get_moves_fn(state)
        if depth == 0 or not moves:
            val = evaluate_fn(state)
            if use_transposition_table:
                tt[state_hash] = (val, depth)
            return val

        if is_max:
            value = float("-inf")
            for move in moves:
                child = apply_move_fn(state, move)
                child_val = _minimax(child, depth - 1, alpha, beta, False)
                value = max(value, child_val)
                if use_alpha_beta:
                    alpha = max(alpha, value)
                    if alpha >= beta:
                        break  # β cutoff
        else:
            value = float("inf")
            for move in moves:
                child = apply_move_fn(state, move)
                child_val = _minimax(child, depth - 1, alpha, beta, True)
                value = min(value, child_val)
                if use_alpha_beta:
                    beta = min(beta, value)
                    if alpha >= beta:
                        break  # α cutoff

        if use_transposition_table:
            tt[state_hash] = (value, depth)
        return value

    # Find the best move at the root
    moves = get_moves_fn(initial_state)
    if not moves:
        elapsed = time.perf_counter() - t0
        return SolverResult(
            solution=np.array([]),
            energy=evaluate_fn(initial_state),
            method=SolverMethod.MINIMAX,
            iterations=nodes_searched[0],
            time_seconds=elapsed,
            converged=True,
        )

    best_move_idx = 0
    if maximizing:
        best_val = float("-inf")
        for i, move in enumerate(moves):
            child = apply_move_fn(initial_state, move)
            val = _minimax(child, max_depth - 1, float("-inf"), float("inf"), False)
            if val > best_val:
                best_val = val
                best_move_idx = i
    else:
        best_val = float("inf")
        for i, move in enumerate(moves):
            child = apply_move_fn(initial_state, move)
            val = _minimax(child, max_depth - 1, float("-inf"), float("inf"), True)
            if val < best_val:
                best_val = val
                best_move_idx = i

    elapsed = time.perf_counter() - t0

    return SolverResult(
        solution=np.array([best_move_idx]),
        energy=best_val,
        method=SolverMethod.MINIMAX,
        iterations=nodes_searched[0],
        time_seconds=elapsed,
        converged=True,
        metadata={
            "best_move": best_move_idx,
            "nodes_searched": nodes_searched[0],
            "transposition_table_size": len(tt),
            "max_depth": max_depth,
        },
    )


def iterative_deepening_minimax(
    evaluate_fn: Callable[[Any], float],
    get_moves_fn: Callable[[Any], List[Any]],
    apply_move_fn: Callable[[Any, Any], Any],
    initial_state: Any,
    max_depth: int = 20,
    time_limit: float = 5.0,
    maximizing: bool = True,
) -> SolverResult:
    """Iterative deepening minimax with time limit.

    Searches at increasing depths until the time limit is reached.
    Returns the best move found so far.
    """
    t0 = time.perf_counter()
    best_result = None

    for depth in range(1, max_depth + 1):
        if time.perf_counter() - t0 > time_limit:
            break
        result = minimax_solve(
            evaluate_fn, get_moves_fn, apply_move_fn,
            initial_state, max_depth=depth, maximizing=maximizing,
        )
        best_result = result
        best_result.metadata["completed_depth"] = depth

    if best_result is not None:
        best_result.time_seconds = time.perf_counter() - t0
    return best_result
