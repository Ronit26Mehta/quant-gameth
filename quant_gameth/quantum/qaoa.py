"""
Quantum Approximate Optimization Algorithm (QAOA).

Follows Mathematical Foundations §E:
    |ψ(γ,β)⟩ = e^{-iβₚĤ_B} e^{-iγₚĤ_C} … e^{-iβ₁Ĥ_B} e^{-iγ₁Ĥ_C} |+⟩^⊗n

The cost Hamiltonian Ĥ_C is diagonal in the computational basis (Ising form).
The mixer Hamiltonian Ĥ_B = ΣᵢXᵢ (bit-flip mixer).

Optimization of (γ, β) uses SciPy optimizers (COBYLA, Nelder-Mead).
"""

from __future__ import annotations

import math
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize as scipy_minimize

from quant_gameth._types import SolverResult, SolverMethod
from quant_gameth.quantum.gates import Gates


def qaoa_solve(
    cost_diagonal: np.ndarray,
    n_qubits: int,
    p_layers: int = 2,
    optimizer: str = "COBYLA",
    max_iter: int = 200,
    initial_params: Optional[np.ndarray] = None,
    seed: int = 42,
    warm_start_bitstring: Optional[str] = None,
) -> SolverResult:
    """Solve a diagonal cost Hamiltonian with QAOA.

    Parameters
    ----------
    cost_diagonal : np.ndarray
        Diagonal of the cost Hamiltonian Ĥ_C, shape ``(2^n,)``.
        Entry ``cost_diagonal[x]`` is the energy of basis state |x⟩.
    n_qubits : int
        Number of qubits.
    p_layers : int
        Number of QAOA layers (circuit depth parameter *p*).
    optimizer : str
        Classical optimizer (``'COBYLA'``, ``'Nelder-Mead'``, ``'Powell'``).
    max_iter : int
        Maximum optimiser iterations.
    initial_params : np.ndarray or None
        Initial (γ, β) values.  Shape ``(2*p,)``.
    seed : int
        RNG seed.
    warm_start_bitstring : str or None
        If given, initialise from this classical solution instead of |+⟩^⊗n.

    Returns
    -------
    SolverResult
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    dim = 1 << n_qubits

    if cost_diagonal.shape != (dim,):
        raise ValueError(f"cost_diagonal must have shape ({dim},)")

    # Precompute mixer matrix elements for each qubit
    # For bit-flip mixer Ĥ_B = ΣᵢXᵢ, the operator e^{-iβΣXᵢ} = ⊗ᵢ e^{-iβXᵢ}
    # e^{-iβX} = cos(β)I - i sin(β)X

    history: List[float] = []

    def build_state(params: np.ndarray) -> np.ndarray:
        """Construct QAOA state for given parameters."""
        gamma = params[:p_layers]
        beta = params[p_layers:]

        # Initial state
        if warm_start_bitstring is not None:
            state = np.zeros(dim, dtype=np.complex128)
            idx = int(warm_start_bitstring, 2)
            state[idx] = 1.0
        else:
            state = np.ones(dim, dtype=np.complex128) / math.sqrt(dim)

        for layer in range(p_layers):
            # Problem unitary: e^{-iγĤ_C} — diagonal, so element-wise
            state *= np.exp(-1j * gamma[layer] * cost_diagonal)

            # Mixer unitary: e^{-iβΣᵢXᵢ} = ⊗ᵢ e^{-iβXᵢ}
            # Apply qubit by qubit using stride-based method
            c_b = math.cos(beta[layer])
            s_b = math.sin(beta[layer])
            for q in range(n_qubits):
                stride = 1 << q
                for block in range(0, dim, 2 * stride):
                    for i in range(block, block + stride):
                        i0 = i
                        i1 = i + stride
                        a0, a1 = state[i0], state[i1]
                        state[i0] = c_b * a0 - 1j * s_b * a1
                        state[i1] = -1j * s_b * a0 + c_b * a1

        return state

    def objective(params: np.ndarray) -> float:
        """⟨ψ(γ,β)|Ĥ_C|ψ(γ,β)⟩"""
        state = build_state(params)
        probs = np.abs(state) ** 2
        val = float(np.dot(probs, cost_diagonal))
        history.append(val)
        return val

    # Initial parameters
    if initial_params is None:
        initial_params = rng.uniform(0, 2 * math.pi, 2 * p_layers)
    else:
        initial_params = np.asarray(initial_params, dtype=float)

    # Optimise
    result = scipy_minimize(
        objective,
        initial_params,
        method=optimizer,
        options={"maxiter": max_iter},
    )

    # Extract solution
    optimal_state = build_state(result.x)
    probs = np.abs(optimal_state) ** 2
    best_idx = int(np.argmax(probs))
    best_bitstring = format(best_idx, f"0{n_qubits}b")
    best_energy = float(cost_diagonal[best_idx])

    elapsed = time.perf_counter() - t0

    return SolverResult(
        solution=np.array([int(b) for b in best_bitstring]),
        energy=best_energy,
        method=SolverMethod.QAOA,
        iterations=result.nfev,
        time_seconds=elapsed,
        converged=result.success,
        constraint_violations=0,
        history=history,
        metadata={
            "best_bitstring": best_bitstring,
            "best_probability": float(probs[best_idx]),
            "optimal_params": result.x.tolist(),
            "p_layers": p_layers,
            "optimizer": optimizer,
            "fun": float(result.fun),
            "expectation_value": float(result.fun),
        },
    )


def build_maxcut_cost_diagonal(
    adjacency: np.ndarray, n_qubits: int
) -> np.ndarray:
    """Build QAOA cost diagonal for MaxCut problem.

    From Math Foundations §D.4:
        Ĥ_C = -Σ_{(i,j)∈E} (I - ZᵢZⱼ)/2
        = -Σ edges that are cut

    Parameters
    ----------
    adjacency : np.ndarray
        Adjacency matrix (symmetric, binary or weighted).
    n_qubits : int

    Returns
    -------
    np.ndarray
        Cost diagonal of length ``2^n_qubits``.
    """
    dim = 1 << n_qubits
    diag = np.zeros(dim)
    for x in range(dim):
        cut = 0.0
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                if adjacency[i, j] != 0:
                    xi = (x >> i) & 1
                    xj = (x >> j) & 1
                    if xi != xj:
                        cut += adjacency[i, j]
        diag[x] = -cut  # negative so minimization finds max cut
    return diag


def build_ising_cost_diagonal(
    h: np.ndarray, J: np.ndarray, n_qubits: int
) -> np.ndarray:
    """Build cost diagonal from Ising model H = Σ Jᵢⱼ σᵢσⱼ + Σ hᵢσᵢ.

    Parameters
    ----------
    h : np.ndarray
        Linear coefficients, shape ``(n_qubits,)``.
    J : np.ndarray
        Coupling matrix, shape ``(n_qubits, n_qubits)``.
    n_qubits : int

    Returns
    -------
    np.ndarray
        Cost diagonal of length ``2^n_qubits``.
    """
    dim = 1 << n_qubits
    diag = np.zeros(dim)
    for x in range(dim):
        spins = np.array([2 * ((x >> q) & 1) - 1 for q in range(n_qubits)],
                         dtype=float)
        energy = float(h @ spins)
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                energy += J[i, j] * spins[i] * spins[j]
        diag[x] = energy
    return diag
