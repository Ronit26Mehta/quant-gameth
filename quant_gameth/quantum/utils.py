"""
Quantum utility functions — fidelity, entropy, Bloch coordinates,
numerical stability helpers.

From Mathematical Foundations §H.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple


def fidelity(sv1: np.ndarray, sv2: np.ndarray) -> float:
    """State fidelity |⟨ψ₁|ψ₂⟩|² for pure states."""
    return float(abs(np.vdot(sv1, sv2)) ** 2)


def mixed_state_fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Fidelity F(ρ,σ) = (Tr √(√ρ σ √ρ))² for density matrices."""
    sqrt_rho = _matrix_sqrt(rho)
    product = sqrt_rho @ sigma @ sqrt_rho
    sqrt_product = _matrix_sqrt(product)
    return float(np.real(np.trace(sqrt_product)) ** 2)


def trace_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Trace distance ½‖ρ - σ‖₁."""
    diff = rho - sigma
    eigenvalues = np.linalg.eigvalsh(diff)
    return float(0.5 * np.sum(np.abs(eigenvalues)))


def von_neumann_entropy(rho: np.ndarray) -> float:
    """S(ρ) = -Tr(ρ log₂ ρ) — von Neumann entropy."""
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > 1e-15]
    return float(-np.sum(eigenvalues * np.log2(eigenvalues)))


def mutual_information(rho_ab: np.ndarray, n_a: int, n_b: int) -> float:
    """Quantum mutual information I(A:B) = S(A) + S(B) - S(AB).

    From §A.4:  I(A:B) = S(ρ_A) + S(ρ_B) - S(ρ_AB)
    """
    dim_a = 1 << n_a
    dim_b = 1 << n_b

    # Partial traces
    rho_a = np.trace(rho_ab.reshape(dim_a, dim_b, dim_a, dim_b), axis1=1, axis2=3)
    rho_b = np.trace(rho_ab.reshape(dim_a, dim_b, dim_a, dim_b), axis1=0, axis2=2)

    s_a = von_neumann_entropy(rho_a)
    s_b = von_neumann_entropy(rho_b)
    s_ab = von_neumann_entropy(rho_ab)

    return s_a + s_b - s_ab


def bloch_vector(sv: np.ndarray) -> np.ndarray:
    """Bloch sphere coordinates (x, y, z) for single-qubit state."""
    rho = np.outer(sv, np.conj(sv))
    sx = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    sz = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    x = float(np.real(np.trace(rho @ sx)))
    y = float(np.real(np.trace(rho @ sy)))
    z = float(np.real(np.trace(rho @ sz)))
    return np.array([x, y, z])


def normalize_state(state: np.ndarray) -> np.ndarray:
    """Renormalise statevector (§H.1)."""
    norm = np.linalg.norm(state)
    if norm < 1e-14:
        raise ValueError("State has collapsed to zero")
    return state / norm


def fix_global_phase(state: np.ndarray) -> np.ndarray:
    """Fix global phase so first non-zero element is real positive (§H.1)."""
    idx = np.argmax(np.abs(state))
    phase = np.angle(state[idx])
    return state * np.exp(-1j * phase)


def purity(rho: np.ndarray) -> float:
    """Tr(ρ²) — purity of a density matrix.  1 for pure, 1/d for maximally mixed."""
    return float(np.real(np.trace(rho @ rho)))


def concurrence_from_density(rho: np.ndarray) -> float:
    """Concurrence for 2-qubit density matrix (§A.4)."""
    sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    sy_sy = np.kron(sy, sy)
    rho_tilde = sy_sy @ np.conj(rho) @ sy_sy
    product = rho @ rho_tilde
    eigvals = np.sort(np.real(np.sqrt(
        np.maximum(np.linalg.eigvals(product), 0)
    )))[::-1]
    return float(max(0.0, eigvals[0] - eigvals[1] - eigvals[2] - eigvals[3]))


def _matrix_sqrt(m: np.ndarray) -> np.ndarray:
    """Matrix square root via eigendecomposition."""
    eigenvalues, eigenvectors = np.linalg.eigh(m)
    eigenvalues = np.maximum(eigenvalues, 0)
    return eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.conj().T
