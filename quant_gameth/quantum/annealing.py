"""
Simulated quantum annealing — classical simulation with transverse field.

Implements both:
1. Classical simulated annealing (Metropolis–Hastings)
2. Simulated quantum annealing (SQA) with path-integral Monte Carlo
3. Parallel tempering for improved exploration

Temperature schedules: linear, exponential, adaptive.
"""

from __future__ import annotations

import math
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from quant_gameth._types import SolverResult, SolverMethod


def simulated_quantum_annealing(
    cost_function: Callable[[np.ndarray], float],
    n_variables: int,
    n_steps: int = 5000,
    n_replicas: int = 8,
    initial_temperature: float = 10.0,
    final_temperature: float = 0.01,
    transverse_field_initial: float = 5.0,
    transverse_field_final: float = 0.001,
    schedule: str = "linear",
    seed: int = 42,
) -> SolverResult:
    """Simulated quantum annealing for binary optimisation.

    Approximates quantum annealing using path-integral Monte Carlo:
    multiple classical replicas coupled by the transverse field.

    Parameters
    ----------
    cost_function : callable
        ``cost_function(x: np.ndarray[0/1]) -> float`` — energy to minimise.
    n_variables : int
        Number of binary variables.
    n_steps : int
        Number of annealing steps.
    n_replicas : int
        Number of Trotter slices (more → better quantum approximation).
    initial_temperature, final_temperature : float
        Temperature schedule endpoints.
    transverse_field_initial, transverse_field_final : float
        Transverse field Γ schedule (drives quantum tunnelling).
    schedule : str
        ``'linear'``, ``'exponential'``, or ``'adaptive'``.
    seed : int
        RNG seed.

    Returns
    -------
    SolverResult
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)

    # Initialise replicas randomly
    replicas = rng.integers(0, 2, size=(n_replicas, n_variables)).astype(np.int8)
    energies = np.array([cost_function(r) for r in replicas])

    best_solution = replicas[np.argmin(energies)].copy()
    best_energy = float(np.min(energies))
    history: List[float] = []

    for step in range(n_steps):
        # Schedule
        frac = step / max(n_steps - 1, 1)

        if schedule == "exponential":
            temp = initial_temperature * (final_temperature / initial_temperature) ** frac
            gamma = transverse_field_initial * (transverse_field_final / transverse_field_initial) ** frac
        elif schedule == "adaptive":
            temp = initial_temperature / (1.0 + step * 0.01)
            gamma = transverse_field_initial * (1.0 - frac)
        else:  # linear
            temp = initial_temperature + (final_temperature - initial_temperature) * frac
            gamma = transverse_field_initial + (transverse_field_final - transverse_field_initial) * frac

        beta = 1.0 / max(temp, 1e-12)

        # Coupling strength between replicas (from transverse field)
        J_perp = -0.5 * temp * math.log(math.tanh(gamma * beta / n_replicas)) if gamma > 1e-12 else 0.0

        for replica_idx in range(n_replicas):
            # Pick random variable to flip
            var = rng.integers(0, n_variables)
            old_spin = replicas[replica_idx, var]
            new_spin = 1 - old_spin

            # Classical energy change
            new_config = replicas[replica_idx].copy()
            new_config[var] = new_spin
            delta_e = cost_function(new_config) - energies[replica_idx]

            # Inter-replica coupling change
            if n_replicas > 1:
                prev = (replica_idx - 1) % n_replicas
                nxt = (replica_idx + 1) % n_replicas
                old_coupling = -J_perp * (
                    (2 * old_spin - 1) * (2 * replicas[prev, var] - 1) +
                    (2 * old_spin - 1) * (2 * replicas[nxt, var] - 1)
                )
                new_coupling = -J_perp * (
                    (2 * new_spin - 1) * (2 * replicas[prev, var] - 1) +
                    (2 * new_spin - 1) * (2 * replicas[nxt, var] - 1)
                )
                delta_e += (new_coupling - old_coupling)

            # Metropolis acceptance
            if delta_e < 0 or rng.random() < math.exp(-beta * delta_e):
                replicas[replica_idx, var] = new_spin
                energies[replica_idx] += cost_function(new_config) - (energies[replica_idx] - delta_e + cost_function(new_config))
                energies[replica_idx] = cost_function(replicas[replica_idx])

                if energies[replica_idx] < best_energy:
                    best_energy = float(energies[replica_idx])
                    best_solution = replicas[replica_idx].copy()

        history.append(best_energy)

    elapsed = time.perf_counter() - t0

    return SolverResult(
        solution=best_solution.astype(float),
        energy=best_energy,
        method=SolverMethod.ANNEALING,
        iterations=n_steps,
        time_seconds=elapsed,
        converged=True,
        history=history,
        metadata={
            "n_replicas": n_replicas,
            "schedule": schedule,
            "best_bitstring": "".join(str(int(b)) for b in best_solution),
        },
    )


def classical_simulated_annealing(
    cost_function: Callable[[np.ndarray], float],
    n_variables: int,
    n_steps: int = 10000,
    initial_temperature: float = 10.0,
    final_temperature: float = 0.001,
    schedule: str = "exponential",
    seed: int = 42,
) -> SolverResult:
    """Classical simulated annealing (no quantum replicas).

    Parameters
    ----------
    cost_function, n_variables, n_steps, initial_temperature,
    final_temperature, schedule, seed : see ``simulated_quantum_annealing``.

    Returns
    -------
    SolverResult
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)

    current = rng.integers(0, 2, size=n_variables).astype(np.int8)
    current_energy = cost_function(current)
    best = current.copy()
    best_energy = current_energy
    history: List[float] = []

    for step in range(n_steps):
        frac = step / max(n_steps - 1, 1)
        if schedule == "exponential":
            temp = initial_temperature * (final_temperature / initial_temperature) ** frac
        else:
            temp = initial_temperature + (final_temperature - initial_temperature) * frac

        # Flip a random bit
        var = rng.integers(0, n_variables)
        candidate = current.copy()
        candidate[var] = 1 - candidate[var]
        candidate_energy = cost_function(candidate)
        delta = candidate_energy - current_energy

        if delta < 0 or rng.random() < math.exp(-delta / max(temp, 1e-12)):
            current = candidate
            current_energy = candidate_energy

        if current_energy < best_energy:
            best_energy = current_energy
            best = current.copy()

        history.append(best_energy)

    elapsed = time.perf_counter() - t0

    return SolverResult(
        solution=best.astype(float),
        energy=best_energy,
        method=SolverMethod.CLASSICAL_SA,
        iterations=n_steps,
        time_seconds=elapsed,
        converged=True,
        history=history,
        metadata={
            "schedule": schedule,
            "best_bitstring": "".join(str(int(b)) for b in best),
        },
    )


def parallel_tempering(
    cost_function: Callable[[np.ndarray], float],
    n_variables: int,
    n_steps: int = 5000,
    n_temperatures: int = 8,
    temp_min: float = 0.01,
    temp_max: float = 20.0,
    seed: int = 42,
) -> SolverResult:
    """Parallel tempering (replica exchange Monte Carlo).

    Runs multiple SA chains at different temperatures with periodic swaps.
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)

    # Geometric temperature ladder
    temps = np.geomspace(temp_min, temp_max, n_temperatures)
    chains = [rng.integers(0, 2, size=n_variables).astype(np.int8)
              for _ in range(n_temperatures)]
    energies = [cost_function(c) for c in chains]

    best = chains[0].copy()
    best_energy = min(energies)
    history: List[float] = []

    for step in range(n_steps):
        # Local moves
        for t_idx in range(n_temperatures):
            var = rng.integers(0, n_variables)
            candidate = chains[t_idx].copy()
            candidate[var] = 1 - candidate[var]
            delta = cost_function(candidate) - energies[t_idx]
            if delta < 0 or rng.random() < math.exp(-delta / max(temps[t_idx], 1e-12)):
                chains[t_idx] = candidate
                energies[t_idx] += delta

        # Replica exchange (adjacent temperatures)
        for t_idx in range(n_temperatures - 1):
            beta_lo = 1.0 / max(temps[t_idx], 1e-12)
            beta_hi = 1.0 / max(temps[t_idx + 1], 1e-12)
            delta = (beta_lo - beta_hi) * (energies[t_idx + 1] - energies[t_idx])
            if delta < 0 or rng.random() < math.exp(-delta):
                chains[t_idx], chains[t_idx + 1] = chains[t_idx + 1], chains[t_idx]
                energies[t_idx], energies[t_idx + 1] = energies[t_idx + 1], energies[t_idx]

        current_best = min(energies)
        if current_best < best_energy:
            best_energy = current_best
            best = chains[energies.index(current_best)].copy()
        history.append(best_energy)

    elapsed = time.perf_counter() - t0

    return SolverResult(
        solution=best.astype(float),
        energy=best_energy,
        method=SolverMethod.ANNEALING,
        iterations=n_steps,
        time_seconds=elapsed,
        converged=True,
        history=history,
        metadata={
            "method": "parallel_tempering",
            "n_temperatures": n_temperatures,
            "best_bitstring": "".join(str(int(b)) for b in best),
        },
    )
