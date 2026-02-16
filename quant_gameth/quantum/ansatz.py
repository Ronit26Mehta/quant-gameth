"""
Ansatz library for variational quantum algorithms.

From Tech Spec §3.3:
  1. Hardware-Efficient Ansatz (HEA): RY-CNOT-RZ layers
  2. QAOA standard ansatz: problem + mixer Hamiltonians
  3. QAOA+ (warm-start, custom mixers)
  4. Unitary Coupled Cluster (UCC) for resource allocation
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

from quant_gameth.quantum.circuit import QuantumCircuit
from quant_gameth.quantum.gates import Gates


def hardware_efficient_ansatz(
    params: np.ndarray,
    n_qubits: int,
    n_layers: int = 2,
    entangling: str = "linear",
) -> QuantumCircuit:
    """Hardware-Efficient Ansatz (§3.3.1).

    Structure per layer: RY(θ) ⊗ ... ⊗ RY(θ) → CNOT chain → RZ(φ) ⊗ ... ⊗ RZ(φ)

    Parameters
    ----------
    params : np.ndarray
        Parameter vector, length ``2 * n_qubits * n_layers``.
    n_qubits : int
    n_layers : int
    entangling : str
        ``'linear'``, ``'circular'``, or ``'full'``.
    """
    circ = QuantumCircuit(n_qubits)
    idx = 0
    for layer in range(n_layers):
        for q in range(n_qubits):
            circ.ry(float(params[idx]), q)
            idx += 1

        if entangling == "linear":
            for q in range(n_qubits - 1):
                circ.cx(q, q + 1)
        elif entangling == "circular":
            for q in range(n_qubits - 1):
                circ.cx(q, q + 1)
            if n_qubits > 2:
                circ.cx(n_qubits - 1, 0)
        elif entangling == "full":
            for q1 in range(n_qubits):
                for q2 in range(q1 + 1, n_qubits):
                    circ.cx(q1, q2)

        for q in range(n_qubits):
            circ.rz(float(params[idx]), q)
            idx += 1
    return circ


def qaoa_ansatz(
    gamma: np.ndarray,
    beta: np.ndarray,
    n_qubits: int,
    cost_pairs: List[tuple],
    cost_weights: Optional[List[float]] = None,
) -> QuantumCircuit:
    """Standard QAOA ansatz (§3.3.2).

    |ψ(γ,β)⟩ = Πₚ U(B,βₚ) U(C,γₚ) |+⟩^⊗n

    Parameters
    ----------
    gamma, beta : np.ndarray
        Parameters, one per layer.
    n_qubits : int
    cost_pairs : list of (int, int)
        Edges (i, j) in the cost Hamiltonian: Ĥ_C = -Σ ZᵢZⱼ/2.
    cost_weights : list of float or None
        Weights for each edge.
    """
    p = len(gamma)
    if cost_weights is None:
        cost_weights = [1.0] * len(cost_pairs)

    circ = QuantumCircuit(n_qubits)

    # Initial |+⟩^⊗n
    for q in range(n_qubits):
        circ.h(q)

    for layer in range(p):
        # Problem unitary: e^{-iγ Σ wᵢⱼ ZᵢZⱼ}
        for (i, j), w in zip(cost_pairs, cost_weights):
            circ.rzz(2 * gamma[layer] * w, i, j)

        # Mixer: e^{-iβ Σ Xᵢ} = ⊗ᵢ Rx(2β)
        for q in range(n_qubits):
            circ.rx(2 * beta[layer], q)

    return circ


def qaoa_plus_ansatz(
    gamma: np.ndarray,
    beta: np.ndarray,
    n_qubits: int,
    cost_pairs: List[tuple],
    mixer_type: str = "xy",
    warm_start: Optional[str] = None,
) -> QuantumCircuit:
    """QAOA+ with custom mixers (§3.3.3).

    Parameters
    ----------
    mixer_type : str
        ``'standard'`` (X mixer), ``'xy'`` (number-conserving),
        ``'grover'`` (Grover mixer for constraint satisfaction).
    warm_start : str or None
        Bit-string for warm-start initialisation.
    """
    p = len(gamma)
    circ = QuantumCircuit(n_qubits)

    # Initialisation
    if warm_start is not None:
        for q in range(n_qubits):
            if warm_start[q] == "1":
                circ.x(q)
    else:
        for q in range(n_qubits):
            circ.h(q)

    for layer in range(p):
        # Cost layer
        for i, j in cost_pairs:
            circ.rzz(2 * gamma[layer], i, j)

        # Mixer layer
        if mixer_type == "xy":
            # XY mixer: preserves Hamming weight
            for q in range(n_qubits - 1):
                circ.rxx(beta[layer], q, q + 1)
                circ.custom(Gates.RYY(beta[layer]), (q, q + 1), "RYY")
        elif mixer_type == "grover":
            # Grover mixer: 2|ψ₀⟩⟨ψ₀| - I
            for q in range(n_qubits):
                circ.h(q)
            for q in range(n_qubits):
                circ.rz(2 * beta[layer], q)
            for q in range(n_qubits):
                circ.h(q)
        else:  # standard
            for q in range(n_qubits):
                circ.rx(2 * beta[layer], q)

    return circ


def ucc_ansatz(
    params: np.ndarray,
    n_qubits: int,
    excitations: Optional[List[tuple]] = None,
) -> QuantumCircuit:
    """Unitary Coupled Cluster ansatz (§3.3.4).

    U(θ) = e^{T(θ) - T†(θ)} with single and double excitations.
    Approximated using first-order Trotter expansion.

    Parameters
    ----------
    params : np.ndarray
        One parameter per excitation.
    n_qubits : int
    excitations : list of tuple or None
        Pairs (i, j) for single excitations or ((i,j), (k,l)) for doubles.
        If None, generates all nearest-neighbor single excitations.
    """
    circ = QuantumCircuit(n_qubits)

    # Initialise reference state (half-filled)
    n_occ = n_qubits // 2
    for q in range(n_occ):
        circ.x(q)

    if excitations is None:
        excitations = [(q, q + 1) for q in range(n_qubits - 1)]

    idx = 0
    for exc in excitations:
        if idx >= len(params):
            break
        theta = float(params[idx])
        if isinstance(exc, tuple) and len(exc) == 2:
            i, j = exc
            # Single excitation: e^{θ(a†ᵢaⱼ - a†ⱼaᵢ)}
            # Jordan-Wigner: CNOT ladder + RY rotation
            circ.cx(i, j)
            circ.ry(2 * theta, j)
            circ.cx(i, j)
        idx += 1

    return circ
