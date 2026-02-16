"""
Evolutionary game theory — replicator dynamics, Moran process, ESS detection.

From Mathematical Foundations §B.4:
    ẋᵢ = xᵢ [u(eᵢ,x) - u(x,x)]
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from quant_gameth._types import EquilibriumResult, SolverMethod


def replicator_dynamics(
    payoff_matrix: np.ndarray,
    initial_population: Optional[np.ndarray] = None,
    dt: float = 0.01,
    n_steps: int = 5000,
    convergence_tol: float = 1e-8,
    seed: int = 42,
) -> EquilibriumResult:
    """Simulate continuous-time replicator dynamics (§B.4).

    ẋᵢ = xᵢ [u(eᵢ, x) - u(x, x)]

    Parameters
    ----------
    payoff_matrix : np.ndarray
        Symmetric payoff matrix A, shape ``(n, n)``.
    initial_population : np.ndarray or None
        Initial strategy frequencies (sums to 1).
    dt : float
        Time step.
    n_steps : int
        Maximum number of steps.
    convergence_tol : float
        Stop when ‖ẋ‖ < tol.
    seed : int

    Returns
    -------
    EquilibriumResult
    """
    t0 = time.perf_counter()
    n = len(payoff_matrix)
    rng = np.random.default_rng(seed)

    if initial_population is None:
        x = rng.dirichlet(np.ones(n))
    else:
        x = np.array(initial_population, dtype=float)
        x = x / x.sum()

    history: List[np.ndarray] = [x.copy()]

    for step in range(n_steps):
        # Fitness: u(eᵢ, x) = (Ax)ᵢ
        fitness = payoff_matrix @ x
        # Average fitness: u(x, x) = x · Ax
        avg_fitness = float(x @ fitness)
        # Replicator equation
        dx = x * (fitness - avg_fitness)
        x = x + dt * dx
        # Project back to simplex (handle numerical errors)
        x = np.maximum(x, 0)
        total = x.sum()
        if total > 1e-14:
            x /= total
        history.append(x.copy())

        if np.linalg.norm(dx) < convergence_tol:
            break

    elapsed = time.perf_counter() - t0

    return EquilibriumResult(
        strategies=[x],
        payoffs=np.array([float(x @ payoff_matrix @ x)]),
        equilibrium_type="evolutionary_stable" if _is_ess(payoff_matrix, x) else "stable_fixed_point",
        method=SolverMethod.REPLICATOR,
        time_seconds=elapsed,
        metadata={
            "n_steps": step + 1,
            "trajectory": [h.tolist() for h in history[::max(1, len(history) // 50)]],
        },
    )


def moran_process(
    payoff_matrix: np.ndarray,
    population_size: int = 100,
    initial_counts: Optional[np.ndarray] = None,
    n_generations: int = 10000,
    mutation_rate: float = 0.001,
    seed: int = 42,
) -> EquilibriumResult:
    """Moran process — stochastic finite-population evolutionary dynamics.

    At each step:
    1. Select an individual proportional to fitness (selection)
    2. Replace a random individual (death)
    3. Apply mutation with probability μ
    """
    t0 = time.perf_counter()
    n_strategies = len(payoff_matrix)
    rng = np.random.default_rng(seed)

    if initial_counts is None:
        counts = np.full(n_strategies, population_size // n_strategies)
        remainder = population_size - counts.sum()
        counts[0] += remainder
    else:
        counts = np.array(initial_counts, dtype=int)

    history: List[np.ndarray] = [counts.copy()]

    for gen in range(n_generations):
        # Current frequencies
        freq = counts / counts.sum()

        # Fitness for each strategy
        fitness = payoff_matrix @ freq
        # Shift to positive for selection probabilities
        fitness_adj = fitness - fitness.min() + 1.0

        # Selection: pick individual to reproduce
        individual_fitness = np.repeat(fitness_adj, counts)
        total_fitness = individual_fitness.sum()
        if total_fitness < 1e-14:
            break
        probs = individual_fitness / total_fitness
        birth_idx = rng.choice(len(individual_fitness), p=probs)

        # Map back to strategy type
        cumsum = np.cumsum(counts)
        birth_type = np.searchsorted(cumsum, birth_idx, side="right")
        birth_type = min(birth_type, n_strategies - 1)

        # Death: random individual
        death_idx = rng.integers(0, population_size)
        death_type = np.searchsorted(cumsum, death_idx, side="right")
        death_type = min(death_type, n_strategies - 1)

        # Apply
        counts[birth_type] += 1
        counts[death_type] -= 1
        counts = np.maximum(counts, 0)

        # Mutation
        if rng.random() < mutation_rate:
            from_type = rng.integers(0, n_strategies)
            to_type = rng.integers(0, n_strategies)
            if counts[from_type] > 0:
                counts[from_type] -= 1
                counts[to_type] += 1

        if gen % 100 == 0:
            history.append(counts.copy())

        # Check fixation
        if np.count_nonzero(counts) == 1:
            break

    elapsed = time.perf_counter() - t0
    final_freq = counts / max(counts.sum(), 1)

    return EquilibriumResult(
        strategies=[final_freq],
        payoffs=np.array([float(final_freq @ payoff_matrix @ final_freq)]),
        equilibrium_type="fixation" if np.count_nonzero(counts) == 1 else "polymorphic",
        method=SolverMethod.MORAN,
        time_seconds=elapsed,
        metadata={
            "final_counts": counts.tolist(),
            "n_generations": gen + 1,
            "trajectory": [h.tolist() for h in history],
        },
    )


def detect_ess(payoff_matrix: np.ndarray) -> List[np.ndarray]:
    """Find Evolutionarily Stable Strategies (§B.4).

    Checks each pure strategy and uniform mix for ESS conditions:
        u(σ*, σ*) > u(σ, σ*)  OR
        u(σ*, σ*) = u(σ, σ*) AND u(σ*, σ) > u(σ, σ)
    """
    n = len(payoff_matrix)
    ess_list: List[np.ndarray] = []

    # Check pure strategies
    for i in range(n):
        sigma_star = np.zeros(n)
        sigma_star[i] = 1.0
        if _is_ess(payoff_matrix, sigma_star):
            ess_list.append(sigma_star)

    # Check uniform mix
    uniform = np.ones(n) / n
    if _is_ess(payoff_matrix, uniform):
        ess_list.append(uniform)

    return ess_list


def _is_ess(payoff_matrix: np.ndarray, sigma_star: np.ndarray) -> bool:
    """Check if σ* is an ESS."""
    n = len(payoff_matrix)
    for i in range(n):
        sigma = np.zeros(n)
        sigma[i] = 1.0
        if np.allclose(sigma, sigma_star):
            continue

        u_star_star = float(sigma_star @ payoff_matrix @ sigma_star)
        u_sigma_star = float(sigma @ payoff_matrix @ sigma_star)

        if u_sigma_star > u_star_star + 1e-10:
            return False
        if abs(u_sigma_star - u_star_star) < 1e-10:
            u_star_sigma = float(sigma_star @ payoff_matrix @ sigma)
            u_sigma_sigma = float(sigma @ payoff_matrix @ sigma)
            if u_sigma_sigma >= u_star_sigma - 1e-10:
                return False
    return True


def fitness_landscape(
    payoff_matrix: np.ndarray,
    resolution: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute fitness landscape for a 2-strategy game.

    Returns
    -------
    frequencies : np.ndarray
        Array of population frequencies for strategy 0.
    fitness_values : np.ndarray
        Average fitness at each frequency, shape ``(resolution, 2)``.
    """
    if payoff_matrix.shape[0] != 2:
        raise ValueError("Fitness landscape only for 2-strategy games")

    freqs = np.linspace(0, 1, resolution)
    fitness_vals = np.zeros((resolution, 2))

    for i, f in enumerate(freqs):
        x = np.array([f, 1.0 - f])
        fit = payoff_matrix @ x
        fitness_vals[i] = fit

    return freqs, fitness_vals
