"""
Repeated games — iterated Prisoner's Dilemma and strategy tournaments.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Classic strategies for iterated PD
# ---------------------------------------------------------------------------

def always_cooperate(history_self: List[int], history_opp: List[int]) -> int:
    return 0  # cooperate

def always_defect(history_self: List[int], history_opp: List[int]) -> int:
    return 1  # defect

def tit_for_tat(history_self: List[int], history_opp: List[int]) -> int:
    if not history_opp:
        return 0
    return history_opp[-1]

def suspicious_tft(history_self: List[int], history_opp: List[int]) -> int:
    if not history_opp:
        return 1
    return history_opp[-1]

def grim_trigger(history_self: List[int], history_opp: List[int]) -> int:
    if any(a == 1 for a in history_opp):
        return 1
    return 0

def pavlov(history_self: List[int], history_opp: List[int]) -> int:
    if not history_self:
        return 0
    if history_self[-1] == history_opp[-1]:
        return 0  # cooperate if last round matched
    return 1

def random_strategy(history_self: List[int], history_opp: List[int]) -> int:
    return int(np.random.random() > 0.5)

def tit_for_two_tats(history_self: List[int], history_opp: List[int]) -> int:
    if len(history_opp) < 2:
        return 0
    if history_opp[-1] == 1 and history_opp[-2] == 1:
        return 1
    return 0


BUILTIN_STRATEGIES = {
    "always_cooperate": always_cooperate,
    "always_defect": always_defect,
    "tit_for_tat": tit_for_tat,
    "suspicious_tft": suspicious_tft,
    "grim_trigger": grim_trigger,
    "pavlov": pavlov,
    "random": random_strategy,
    "tit_for_two_tats": tit_for_two_tats,
}


# ---------------------------------------------------------------------------
# Iterated game
# ---------------------------------------------------------------------------

def iterated_game(
    strategy_1: Callable,
    strategy_2: Callable,
    payoff_matrix_1: Optional[np.ndarray] = None,
    payoff_matrix_2: Optional[np.ndarray] = None,
    n_rounds: int = 200,
    discount_factor: float = 1.0,
    noise: float = 0.0,
    seed: int = 42,
) -> Dict:
    """Play an iterated game between two strategies.

    Parameters
    ----------
    strategy_1, strategy_2 : callable
        ``strategy(history_self, history_opp) -> action (int)``
    payoff_matrix_1, payoff_matrix_2 : np.ndarray or None
        2×2 payoff matrices.  Defaults to standard PD.
    n_rounds : int
    discount_factor : float
        Discount factor δ for future payoffs.
    noise : float
        Probability of action flipping (trembling hand).
    seed : int
    """
    rng = np.random.default_rng(seed)

    if payoff_matrix_1 is None:
        payoff_matrix_1 = np.array([[3, 0], [5, 1]])  # PD
    if payoff_matrix_2 is None:
        payoff_matrix_2 = np.array([[3, 5], [0, 1]])

    history_1: List[int] = []
    history_2: List[int] = []
    payoffs_1: List[float] = []
    payoffs_2: List[float] = []

    for t in range(n_rounds):
        a1 = strategy_1(history_1, history_2)
        a2 = strategy_2(history_2, history_1)

        # Trembling hand
        if noise > 0:
            if rng.random() < noise:
                a1 = 1 - a1
            if rng.random() < noise:
                a2 = 1 - a2

        discount = discount_factor ** t
        payoffs_1.append(float(payoff_matrix_1[a1, a2]) * discount)
        payoffs_2.append(float(payoff_matrix_2[a1, a2]) * discount)

        history_1.append(a1)
        history_2.append(a2)

    return {
        "total_payoff_1": sum(payoffs_1),
        "total_payoff_2": sum(payoffs_2),
        "avg_payoff_1": sum(payoffs_1) / n_rounds,
        "avg_payoff_2": sum(payoffs_2) / n_rounds,
        "cooperation_rate_1": 1.0 - np.mean(history_1),
        "cooperation_rate_2": 1.0 - np.mean(history_2),
        "history_1": history_1,
        "history_2": history_2,
    }


# ---------------------------------------------------------------------------
# Round-robin tournament
# ---------------------------------------------------------------------------

def tournament(
    strategies: Optional[Dict[str, Callable]] = None,
    n_rounds: int = 200,
    noise: float = 0.0,
    seed: int = 42,
) -> Dict:
    """Run a round-robin tournament between strategies.

    Parameters
    ----------
    strategies : dict of {name: callable} or None
        Default: all built-in strategies.
    n_rounds : int
        Rounds per match.
    noise : float
    seed : int

    Returns
    -------
    dict
        Rankings, scores, and head-to-head results.
    """
    if strategies is None:
        strategies = BUILTIN_STRATEGIES

    names = list(strategies.keys())
    n = len(names)
    scores = {name: 0.0 for name in names}
    head_to_head: Dict[Tuple[str, str], Dict] = {}

    for i in range(n):
        for j in range(i + 1, n):
            result = iterated_game(
                strategies[names[i]],
                strategies[names[j]],
                n_rounds=n_rounds,
                noise=noise,
                seed=seed + i * n + j,
            )
            scores[names[i]] += result["avg_payoff_1"]
            scores[names[j]] += result["avg_payoff_2"]
            head_to_head[(names[i], names[j])] = {
                "payoff_1": result["avg_payoff_1"],
                "payoff_2": result["avg_payoff_2"],
                "coop_1": result["cooperation_rate_1"],
                "coop_2": result["cooperation_rate_2"],
            }

    ranking = sorted(scores.items(), key=lambda x: -x[1])

    return {
        "ranking": [(name, float(score)) for name, score in ranking],
        "scores": {k: float(v) for k, v in scores.items()},
        "head_to_head": {f"{k[0]}_vs_{k[1]}": v for k, v in head_to_head.items()},
    }
