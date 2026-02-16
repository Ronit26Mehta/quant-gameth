"""
Grover's search algorithm — classical simulation.

Follows Mathematical Foundations §A (amplitude amplification) and §1.2.2:
    G = (2|ψ⟩⟨ψ| - I) · (I - 2|target⟩⟨target|)
    Optimal iterations: ⌊π/4 √(N/M)⌋ where M = #targets
"""

from __future__ import annotations

import math
import time
from typing import Callable, List, Optional, Union

import numpy as np

from quant_gameth._types import SolverResult, SolverMethod


def grover_search(
    n_qubits: int,
    oracle_function: Callable[[int], bool],
    n_targets: int = 1,
    n_iterations: Optional[int] = None,
    seed: int = 42,
) -> SolverResult:
    """Run Grover's algorithm to find marked states.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (search space size = 2^n_qubits).
    oracle_function : callable
        ``oracle_function(x) -> bool``: returns True for target states.
    n_targets : int
        Number of marked/target items (used for optimal iteration count).
    n_iterations : int or None
        Override the optimal iteration count.
    seed : int
        RNG seed for measurement.

    Returns
    -------
    SolverResult
        The most probable solution found.
    """
    t0 = time.perf_counter()
    dim = 1 << n_qubits

    # Build oracle diagonal: -1 for targets, +1 otherwise
    oracle_diag = np.ones(dim, dtype=np.complex128)
    target_indices: List[int] = []
    for x in range(dim):
        if oracle_function(x):
            oracle_diag[x] = -1.0
            target_indices.append(x)

    actual_n_targets = len(target_indices)
    if actual_n_targets == 0:
        raise ValueError("Oracle marks zero states — no solution exists")

    # Optimal number of Grover iterations
    if n_iterations is None:
        n_iterations = max(1, int(math.floor(
            (math.pi / 4.0) * math.sqrt(dim / actual_n_targets)
        )))

    # Uniform superposition |+⟩^⊗n
    state = np.ones(dim, dtype=np.complex128) / math.sqrt(dim)

    # Diffusion operator (2|ψ₀⟩⟨ψ₀| - I) applied to state
    history: List[float] = []

    for iteration in range(n_iterations):
        # Step 1: Oracle — flip amplitude of marked states
        state = state * oracle_diag

        # Step 2: Diffusion — reflect about mean amplitude
        mean_amp = np.mean(state)
        state = 2.0 * mean_amp - state

        # Track probability of finding a target
        target_prob = sum(abs(state[t]) ** 2 for t in target_indices)
        history.append(float(target_prob))

    # Measurement: find the most probable state
    probs = np.abs(state) ** 2
    best_idx = int(np.argmax(probs))
    best_prob = float(probs[best_idx])

    elapsed = time.perf_counter() - t0

    return SolverResult(
        solution=np.array([int(b) for b in format(best_idx, f"0{n_qubits}b")]),
        energy=-best_prob,  # negative so lower is better
        method=SolverMethod.GROVER,
        iterations=n_iterations,
        time_seconds=elapsed,
        converged=oracle_function(best_idx),
        constraint_violations=0 if oracle_function(best_idx) else 1,
        history=history,
        metadata={
            "best_bitstring": format(best_idx, f"0{n_qubits}b"),
            "best_probability": best_prob,
            "n_targets": actual_n_targets,
            "search_space_size": dim,
        },
    )


def grover_oracle_from_constraints(
    n_qubits: int,
    constraint_fn: Callable[[np.ndarray], bool],
) -> Callable[[int], bool]:
    """Create an oracle function from a constraint function on binary arrays.

    Parameters
    ----------
    n_qubits : int
    constraint_fn : callable
        ``constraint_fn(x: np.ndarray[bool]) -> bool``

    Returns
    -------
    callable
        Oracle suitable for ``grover_search``.
    """
    def oracle(x: int) -> bool:
        bits = np.array([(x >> i) & 1 for i in range(n_qubits)], dtype=bool)
        return constraint_fn(bits)
    return oracle
