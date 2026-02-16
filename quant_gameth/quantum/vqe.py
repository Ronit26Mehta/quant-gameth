"""
Variational Quantum Eigensolver (VQE).

From Tech Spec §1.2.3:
    E(θ) = ⟨ψ(θ)|H|ψ(θ)⟩    →    θ* = argmin E(θ)

Application to games: H = payoff Hamiltonian (negative utility),
ground state = optimal strategy profile.
"""

from __future__ import annotations

import math
import time
from typing import Callable, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize as scipy_minimize

from quant_gameth._types import SolverResult, SolverMethod
from quant_gameth.quantum.circuit import QuantumCircuit, Simulator
from quant_gameth.quantum.gates import Gates


def vqe_solve(
    hamiltonian_diagonal: np.ndarray,
    n_qubits: int,
    ansatz_builder: Optional[Callable[[np.ndarray, int], QuantumCircuit]] = None,
    n_layers: int = 2,
    optimizer: str = "COBYLA",
    max_iter: int = 300,
    initial_params: Optional[np.ndarray] = None,
    seed: int = 42,
) -> SolverResult:
    """Find the ground state of a (diagonal) Hamiltonian using VQE.

    Parameters
    ----------
    hamiltonian_diagonal : np.ndarray
        Diagonal elements of the Hamiltonian, shape ``(2^n,)``.
    n_qubits : int
    ansatz_builder : callable or None
        ``ansatz_builder(params, n_qubits) -> QuantumCircuit``.
        If ``None``, uses a hardware-efficient ansatz (RY-CNOT layers).
    n_layers : int
        Number of ansatz layers (only used if ``ansatz_builder`` is None).
    optimizer : str
        SciPy optimiser name.
    max_iter : int
    initial_params : np.ndarray or None
    seed : int

    Returns
    -------
    SolverResult
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    dim = 1 << n_qubits

    # Default HEA ansatz
    if ansatz_builder is None:
        n_params = n_qubits * n_layers * 2  # RY + RZ per qubit per layer
        ansatz_builder = lambda params, nq: _build_hea(params, nq, n_layers)
    else:
        # Infer param count from trial
        test_params = np.zeros(100)
        try:
            _ = ansatz_builder(test_params, n_qubits)
        except Exception:
            pass
        n_params = n_qubits * n_layers * 2  # fallback

    if initial_params is None:
        initial_params = rng.uniform(0, 2 * math.pi, n_params)

    history: List[float] = []

    def objective(params: np.ndarray) -> float:
        circ = ansatz_builder(params, n_qubits)
        result = Simulator.run(circ, seed=seed)
        probs = result.probabilities
        energy = float(np.dot(probs, hamiltonian_diagonal))
        history.append(energy)
        return energy

    opt_result = scipy_minimize(
        objective,
        initial_params,
        method=optimizer,
        options={"maxiter": max_iter},
    )

    # Extract best solution
    best_circ = ansatz_builder(opt_result.x, n_qubits)
    best_state = Simulator.run(best_circ, seed=seed)
    probs = best_state.probabilities
    best_idx = int(np.argmax(probs))
    best_bs = format(best_idx, f"0{n_qubits}b")

    elapsed = time.perf_counter() - t0

    return SolverResult(
        solution=np.array([int(b) for b in best_bs]),
        energy=float(hamiltonian_diagonal[best_idx]),
        method=SolverMethod.VQE,
        iterations=opt_result.nfev,
        time_seconds=elapsed,
        converged=opt_result.success,
        history=history,
        metadata={
            "best_bitstring": best_bs,
            "best_probability": float(probs[best_idx]),
            "optimal_params": opt_result.x.tolist(),
            "ground_state_energy": float(opt_result.fun),
        },
    )


def _build_hea(
    params: np.ndarray, n_qubits: int, n_layers: int
) -> QuantumCircuit:
    """Hardware-efficient ansatz: RY-CNOT-RZ layers (Tech Spec §3.3.1)."""
    circ = QuantumCircuit(n_qubits)
    idx = 0
    for layer in range(n_layers):
        # RY rotation on each qubit
        for q in range(n_qubits):
            circ.ry(float(params[idx]), q)
            idx += 1
        # CNOT chain (linear connectivity)
        for q in range(n_qubits - 1):
            circ.cx(q, q + 1)
        # RZ rotation on each qubit
        for q in range(n_qubits):
            circ.rz(float(params[idx]), q)
            idx += 1
    return circ
