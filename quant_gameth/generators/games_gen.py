"""
Random game generators.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from quant_gameth.games.normal_form import NormalFormGame


def generate_random_game(
    n_strategies_1: int = 3,
    n_strategies_2: int = 3,
    game_type: str = "general",
    seed: int = 42,
) -> NormalFormGame:
    """Generate a random normal-form game.

    Parameters
    ----------
    n_strategies_1, n_strategies_2 : int
    game_type : str
        ``'general'``, ``'zero_sum'``, ``'symmetric'``, ``'potential'``.
    seed : int
    """
    rng = np.random.default_rng(seed)

    if game_type == "zero_sum":
        A = rng.uniform(-5, 5, (n_strategies_1, n_strategies_2))
        return NormalFormGame(A, name="random_zero_sum")

    elif game_type == "symmetric":
        A = rng.uniform(0, 10, (n_strategies_1, n_strategies_1))
        return NormalFormGame(A, A.T, name="random_symmetric")

    elif game_type == "potential":
        potential = rng.uniform(0, 10, (n_strategies_1, n_strategies_2))
        A = np.zeros_like(potential)
        B = np.zeros_like(potential)
        for i in range(n_strategies_1):
            for j in range(n_strategies_2):
                A[i, j] = potential[i, j] + rng.normal(0, 0.5)
                B[i, j] = potential[i, j] + rng.normal(0, 0.5)
        return NormalFormGame(A, B, name="random_potential")

    else:
        A = rng.uniform(0, 10, (n_strategies_1, n_strategies_2))
        B = rng.uniform(0, 10, (n_strategies_1, n_strategies_2))
        return NormalFormGame(A, B, name="random_general")
