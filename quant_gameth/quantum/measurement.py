"""
Measurement utilities — Born rule, shot sampling, partial measurement.

Follows Mathematical Foundations §A.3 and §F.4.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


def measure_statevector(
    sv: np.ndarray,
    n_qubits: int,
    n_shots: int = 1024,
    seed: Optional[int] = 42,
) -> Dict[str, int]:
    """Sample from statevector via Born rule.

    Parameters
    ----------
    sv : np.ndarray
        Complex statevector of length ``2**n_qubits``.
    n_qubits : int
    n_shots : int
        Number of measurement samples.
    seed : int or None
        RNG seed for reproducibility.

    Returns
    -------
    dict
        ``{bitstring: count}``
    """
    rng = np.random.default_rng(seed)
    probs = np.abs(sv) ** 2
    # Normalise if slightly off
    total = probs.sum()
    if abs(total - 1.0) > 1e-10:
        probs = probs / total
    outcomes = rng.choice(len(probs), size=n_shots, p=probs)
    counts: Dict[str, int] = {}
    for o in outcomes:
        bs = format(int(o), f"0{n_qubits}b")
        counts[bs] = counts.get(bs, 0) + 1
    return counts


def probability_distribution(sv: np.ndarray) -> np.ndarray:
    """Return the full probability distribution (deterministic, no sampling)."""
    return np.abs(sv) ** 2


def expectation_value(
    sv: np.ndarray, operator: np.ndarray
) -> float:
    """⟨ψ|O|ψ⟩ for Hermitian operator O.

    Handles both dense (2D) and diagonal (1D) operators.
    """
    if operator.ndim == 1:
        return float(np.real(np.sum(np.abs(sv) ** 2 * operator)))
    return float(np.real(np.vdot(sv, operator @ sv)))


def expectation_ising_diagonal(
    sv: np.ndarray,
    n_qubits: int,
    h: np.ndarray,
    J: np.ndarray,
) -> float:
    """Fast expectation for Ising Hamiltonian H = Σ Jᵢⱼ σᵢσⱼ + Σ hᵢσᵢ.

    Since the Ising Hamiltonian is diagonal in the Z-basis, we compute:
        ⟨H⟩ = Σₓ |⟨x|ψ⟩|² H(x)
    where H(x) = Σᵢⱼ Jᵢⱼ sᵢsⱼ + Σᵢ hᵢsᵢ  with sᵢ = 2xᵢ-1 ∈ {-1,+1}.

    This is O(n² · 2^n) but avoids building 2^n × 2^n matrix.
    """
    dim = 1 << n_qubits
    probs = np.abs(sv) ** 2
    energy = 0.0
    for x in range(dim):
        if probs[x] < 1e-16:
            continue
        spins = np.array([(2 * ((x >> q) & 1) - 1) for q in range(n_qubits)])
        cost = float(h @ spins)
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                if J[i, j] != 0:
                    cost += J[i, j] * spins[i] * spins[j]
        energy += probs[x] * cost
    return energy


def partial_measurement(
    sv: np.ndarray,
    n_qubits: int,
    qubit: int,
    seed: Optional[int] = 42,
) -> Tuple[int, np.ndarray]:
    """Projective measurement of a single qubit, collapsing the state.

    Returns
    -------
    outcome : int
        0 or 1.
    collapsed_sv : np.ndarray
        Post-measurement state (normalised).
    """
    rng = np.random.default_rng(seed)
    dim = 1 << n_qubits
    prob0 = 0.0
    for i in range(dim):
        if (i >> qubit) & 1 == 0:
            prob0 += abs(sv[i]) ** 2
    outcome = 0 if rng.random() < prob0 else 1
    collapsed = np.zeros_like(sv)
    for i in range(dim):
        if ((i >> qubit) & 1) == outcome:
            collapsed[i] = sv[i]
    norm = np.linalg.norm(collapsed)
    if norm > 1e-14:
        collapsed /= norm
    return outcome, collapsed


def marginal_probabilities(
    sv: np.ndarray,
    n_qubits: int,
    qubits: List[int],
) -> np.ndarray:
    """Compute marginal probability distribution over a subset of qubits.

    Returns an array of shape ``(2**len(qubits), )`` with probabilities.
    """
    n_sub = len(qubits)
    dim = 1 << n_qubits
    sub_dim = 1 << n_sub
    probs = np.abs(sv) ** 2
    marginal = np.zeros(sub_dim)
    for x in range(dim):
        sub_idx = 0
        for k, q in enumerate(qubits):
            sub_idx |= ((x >> q) & 1) << k
        marginal[sub_idx] += probs[x]
    return marginal
