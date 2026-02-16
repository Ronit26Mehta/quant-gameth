"""
Cooperative game theory — Shapley value, core, nucleolus.
"""

from __future__ import annotations

import itertools
import math
import time
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np


def shapley_value(
    n_players: int,
    characteristic_function: Callable[[Set[int]], float],
) -> np.ndarray:
    """Compute Shapley values for a cooperative game.

    φᵢ = Σ_{S ⊆ N\\{i}} [|S|!(n-|S|-1)!/n!] [v(S∪{i}) - v(S)]

    Parameters
    ----------
    n_players : int
    characteristic_function : callable
        ``v(S: set of int) -> float`` — value of coalition S.

    Returns
    -------
    np.ndarray
        Shapley value for each player, shape ``(n_players,)``.
    """
    shapley = np.zeros(n_players)
    factorial_n = math.factorial(n_players)

    for i in range(n_players):
        others = [j for j in range(n_players) if j != i]
        for size in range(len(others) + 1):
            for subset in itertools.combinations(others, size):
                s = set(subset)
                s_with_i = s | {i}
                marginal = characteristic_function(s_with_i) - characteristic_function(s)
                weight = math.factorial(len(s)) * math.factorial(n_players - len(s) - 1)
                shapley[i] += weight * marginal / factorial_n

    return shapley


def shapley_value_sampling(
    n_players: int,
    characteristic_function: Callable[[Set[int]], float],
    n_samples: int = 10000,
    seed: int = 42,
) -> np.ndarray:
    """Approximate Shapley values via permutation sampling.

    For large games where exact computation is intractable.
    """
    rng = np.random.default_rng(seed)
    shapley = np.zeros(n_players)

    for _ in range(n_samples):
        perm = rng.permutation(n_players)
        coalition = set()
        prev_val = 0.0
        for player in perm:
            coalition.add(int(player))
            curr_val = characteristic_function(coalition)
            shapley[int(player)] += curr_val - prev_val
            prev_val = curr_val

    return shapley / n_samples


def core(
    n_players: int,
    characteristic_function: Callable[[Set[int]], float],
) -> Tuple[bool, Optional[np.ndarray]]:
    """Check if the core is non-empty and find a core allocation.

    The core contains allocations x such that:
        Σᵢ∈S xᵢ ≥ v(S) for all S ⊆ N
        Σᵢ xᵢ = v(N)

    Uses linear programming.
    """
    from scipy.optimize import linprog

    all_players = set(range(n_players))
    v_grand = characteristic_function(all_players)

    # Variables: x₁, ..., x_n
    # Minimise: 0 (feasibility check)
    c = np.zeros(n_players)

    # Coalition constraints: -Σᵢ∈S xᵢ ≤ -v(S)
    coalitions = []
    for size in range(1, n_players):
        for subset in itertools.combinations(range(n_players), size):
            coalitions.append(set(subset))

    n_constraints = len(coalitions)
    A_ub = np.zeros((n_constraints, n_players))
    b_ub = np.zeros(n_constraints)

    for idx, s in enumerate(coalitions):
        for i in s:
            A_ub[idx, i] = -1.0
        b_ub[idx] = -characteristic_function(s)

    # Efficiency: Σxᵢ = v(N)
    A_eq = np.ones((1, n_players))
    b_eq = np.array([v_grand])

    # Individual rationality: xᵢ ≥ v({i})
    bounds = [
        (characteristic_function({i}), None) for i in range(n_players)
    ]

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method="highs")

    if result.success:
        return True, result.x
    return False, None


def nucleolus(
    n_players: int,
    characteristic_function: Callable[[Set[int]], float],
) -> np.ndarray:
    """Compute the nucleolus — the allocation that lexicographically minimises
    the maximum excess (unhappiness) of any coalition.

    Uses iterative LP.
    """
    from scipy.optimize import linprog

    all_players = set(range(n_players))
    v_grand = characteristic_function(all_players)

    coalitions = []
    for size in range(1, n_players):
        for subset in itertools.combinations(range(n_players), size):
            coalitions.append(set(subset))

    n_coal = len(coalitions)

    # Start with Shapley value as initial guess, then refine
    x = shapley_value(n_players, characteristic_function)

    # Iterative: minimise maximum excess
    for iteration in range(20):
        # Compute excesses: e(S,x) = v(S) - Σᵢ∈S xᵢ
        excesses = []
        for s in coalitions:
            excess = characteristic_function(s) - sum(x[i] for i in s)
            excesses.append(excess)

        max_excess = max(excesses)
        if max_excess < 1e-10:
            break

        # LP to minimise max excess
        # Variables: x₁,...,x_n, t (where t ≥ e(S,x) for all S)
        c_lp = np.zeros(n_players + 1)
        c_lp[-1] = 1.0  # minimise t

        # Constraints: v(S) - Σᵢ∈S xᵢ ≤ t → -Σᵢ∈S xᵢ - t ≤ -v(S)
        A_ub = np.zeros((n_coal, n_players + 1))
        b_ub = np.zeros(n_coal)
        for idx, s in enumerate(coalitions):
            for i in s:
                A_ub[idx, i] = -1.0
            A_ub[idx, -1] = -1.0
            b_ub[idx] = -characteristic_function(s)

        A_eq = np.zeros((1, n_players + 1))
        A_eq[0, :n_players] = 1.0
        b_eq = np.array([v_grand])

        bounds = [(None, None)] * n_players + [(None, None)]

        result = linprog(c_lp, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                         bounds=bounds, method="highs")

        if result.success:
            x = result.x[:n_players]
        else:
            break

    return x


def banzhaf_index(
    n_players: int,
    characteristic_function: Callable[[Set[int]], float],
    threshold: float = 0.5,
) -> np.ndarray:
    """Compute Banzhaf power index.

    Measures how often a player is pivotal (swing voter).
    """
    index = np.zeros(n_players)
    others_list = list(range(n_players))

    for i in range(n_players):
        others = [j for j in others_list if j != i]
        n_pivotal = 0
        n_coalitions = 0

        for size in range(len(others) + 1):
            for subset in itertools.combinations(others, size):
                s = set(subset)
                s_with_i = s | {i}
                if (characteristic_function(s_with_i) >= threshold and
                        characteristic_function(s) < threshold):
                    n_pivotal += 1
                n_coalitions += 1

        index[i] = n_pivotal / max(n_coalitions, 1)

    # Normalise
    total = index.sum()
    if total > 0:
        index /= total
    return index
