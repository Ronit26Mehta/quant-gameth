"""
Quantum state representation — statevector and density matrix.

Follows the Hilbert space formulation from Mathematical Foundations §A.1:
    |ψ⟩ = Σᵢ αᵢ|i⟩   where  Σᵢ|αᵢ|² = 1

Supports n-qubit systems up to ~25 qubits on classical hardware
(2^25 ≈ 33 M complex amplitudes ≈ 512 MB).
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional, Tuple, Union


class QuantumState:
    """N-qubit quantum state stored as a complex statevector.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.  State dimension is ``2**n_qubits``.
    seed : int or None
        Random seed for reproducible measurements.
    """

    __slots__ = ("n_qubits", "dim", "_sv", "_rng")

    def __init__(self, n_qubits: int, seed: Optional[int] = 42):
        if n_qubits < 1:
            raise ValueError("n_qubits must be >= 1")
        self.n_qubits: int = n_qubits
        self.dim: int = 1 << n_qubits
        self._sv: np.ndarray = np.zeros(self.dim, dtype=np.complex128)
        self._sv[0] = 1.0 + 0j  # initialise to |00…0⟩
        self._rng: np.random.Generator = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def statevector(self) -> np.ndarray:
        """Return a *copy* of the internal statevector."""
        return self._sv.copy()

    @statevector.setter
    def statevector(self, sv: np.ndarray) -> None:
        if sv.shape != (self.dim,):
            raise ValueError(
                f"Expected shape ({self.dim},), got {sv.shape}"
            )
        self._sv = np.asarray(sv, dtype=np.complex128)

    @property
    def probabilities(self) -> np.ndarray:
        """Born-rule probabilities |αᵢ|² for each computational basis state."""
        return np.abs(self._sv) ** 2

    @property
    def norm(self) -> float:
        """L-2 norm of the statevector."""
        return float(np.linalg.norm(self._sv))

    # ------------------------------------------------------------------
    # State initialisation helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_statevector(
        cls, sv: np.ndarray, seed: Optional[int] = 42
    ) -> "QuantumState":
        """Create a QuantumState from an existing statevector."""
        n = int(np.log2(len(sv)))
        if 1 << n != len(sv):
            raise ValueError("Statevector length must be a power of 2")
        qs = cls(n, seed=seed)
        qs._sv = np.asarray(sv, dtype=np.complex128)
        return qs

    @classmethod
    def uniform_superposition(
        cls, n_qubits: int, seed: Optional[int] = 42
    ) -> "QuantumState":
        """Create the |+⟩^⊗n state (equal superposition over all basis states)."""
        qs = cls(n_qubits, seed=seed)
        qs._sv[:] = 1.0 / np.sqrt(qs.dim)
        return qs

    @classmethod
    def from_bitstring(
        cls, bitstring: str, seed: Optional[int] = 42
    ) -> "QuantumState":
        """Create a computational basis state from a bit-string like '0110'."""
        n = len(bitstring)
        qs = cls(n, seed=seed)
        idx = int(bitstring, 2)
        qs._sv[idx] = 1.0 + 0j
        return qs

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def normalize(self) -> None:
        """Re-normalise the state (useful after sequences of operations)."""
        n = np.linalg.norm(self._sv)
        if n < 1e-14:
            raise ValueError("State has collapsed to the zero vector")
        self._sv /= n

    def apply_operator(self, operator: np.ndarray) -> None:
        """Apply a (2^n x 2^n) unitary/operator to the full state."""
        self._sv = operator @ self._sv

    def tensor_product(self, other: "QuantumState") -> "QuantumState":
        """Return |self⟩ ⊗ |other⟩ as a new QuantumState."""
        new_sv = np.kron(self._sv, other._sv)
        return QuantumState.from_statevector(new_sv, seed=None)

    # ------------------------------------------------------------------
    # Measurement (Born rule — §A.3)
    # ------------------------------------------------------------------

    def measure_all(self, n_shots: int = 1024) -> dict:
        """Measure all qubits ``n_shots`` times.

        Returns
        -------
        dict
            ``{bitstring: count}`` for each observed outcome.
        """
        probs = self.probabilities
        outcomes = self._rng.choice(self.dim, size=n_shots, p=probs)
        counts: dict = {}
        for o in outcomes:
            bs = format(o, f"0{self.n_qubits}b")
            counts[bs] = counts.get(bs, 0) + 1
        return counts

    def measure_qubit(self, qubit: int) -> Tuple[int, "QuantumState"]:
        """Projectively measure a single qubit, collapsing the state.

        Returns
        -------
        outcome : int
            0 or 1.
        new_state : QuantumState
            Post-measurement (collapsed and normalised) state.
        """
        stride = 1 << qubit
        prob0 = 0.0
        for i in range(self.dim):
            if (i >> qubit) & 1 == 0:
                prob0 += abs(self._sv[i]) ** 2
        outcome = 0 if self._rng.random() < prob0 else 1
        new_sv = np.zeros_like(self._sv)
        for i in range(self.dim):
            if ((i >> qubit) & 1) == outcome:
                new_sv[i] = self._sv[i]
        norm = np.linalg.norm(new_sv)
        if norm > 1e-14:
            new_sv /= norm
        ns = QuantumState.from_statevector(new_sv, seed=None)
        return outcome, ns

    def get_probability_distribution(self) -> np.ndarray:
        """Full probability distribution (deterministic, no sampling)."""
        return self.probabilities

    # ------------------------------------------------------------------
    # Expectation values (§A.3)
    # ------------------------------------------------------------------

    def expectation(self, operator: np.ndarray) -> float:
        """⟨ψ|O|ψ⟩  for Hermitian observable *O*."""
        return float(np.real(np.vdot(self._sv, operator @ self._sv)))

    def expectation_diagonal(self, diagonal: np.ndarray) -> float:
        """Fast expectation for diagonal operators (e.g. Ising Hamiltonians).

        Parameters
        ----------
        diagonal : np.ndarray
            1-D array of length ``dim`` with eigenvalues.
        """
        return float(np.real(np.sum(self.probabilities * diagonal)))

    # ------------------------------------------------------------------
    # Density matrix & entropy (§A.1, §A.4)
    # ------------------------------------------------------------------

    def density_matrix(self) -> np.ndarray:
        """ρ = |ψ⟩⟨ψ|"""
        return np.outer(self._sv, np.conj(self._sv))

    def partial_trace(self, keep_qubits: List[int]) -> np.ndarray:
        """Partial trace over all qubits *not* in *keep_qubits*.

        Returns the reduced density matrix for the subsystem.
        """
        n = self.n_qubits
        rho = self.density_matrix().reshape([2] * (2 * n))
        trace_qubits = sorted(set(range(n)) - set(keep_qubits))
        for q in reversed(trace_qubits):
            rho = np.trace(rho, axis1=q, axis2=q + n)
            n -= 1
        k = len(keep_qubits)
        return rho.reshape(1 << k, 1 << k)

    def von_neumann_entropy(
        self, subsystem: Optional[List[int]] = None
    ) -> float:
        """Von Neumann entropy S(ρ) = -Tr(ρ log₂ ρ).

        Parameters
        ----------
        subsystem : list of int or None
            If given, compute entropy of the reduced density matrix.
            If ``None``, compute entropy of the full state (0 for pure).
        """
        if subsystem is not None:
            rho = self.partial_trace(subsystem)
        else:
            rho = self.density_matrix()
        eigenvalues = np.linalg.eigvalsh(rho)
        eigenvalues = eigenvalues[eigenvalues > 1e-15]
        return float(-np.sum(eigenvalues * np.log2(eigenvalues)))

    def concurrence(self) -> float:
        """Concurrence for 2-qubit states (§A.4).

        C(ρ) = max{0, λ₁ - λ₂ - λ₃ - λ₄}
        """
        if self.n_qubits != 2:
            raise ValueError("Concurrence defined only for 2-qubit states")
        rho = self.density_matrix()
        sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        sy_sy = np.kron(sy, sy)
        rho_tilde = sy_sy @ np.conj(rho) @ sy_sy
        product = rho @ rho_tilde
        eigvals = np.sort(np.real(np.sqrt(np.maximum(
            np.linalg.eigvals(product), 0
        ))))[::-1]
        return float(max(0.0, eigvals[0] - eigvals[1] - eigvals[2] - eigvals[3]))

    # ------------------------------------------------------------------
    # Bloch sphere (single qubit)
    # ------------------------------------------------------------------

    def bloch_vector(self) -> np.ndarray:
        """Bloch sphere coordinates (x, y, z) for a 1-qubit state."""
        if self.n_qubits != 1:
            raise ValueError("Bloch vector defined for single-qubit states")
        rho = self.density_matrix()
        sx = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        sz = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        x = float(np.real(np.trace(rho @ sx)))
        y = float(np.real(np.trace(rho @ sy)))
        z = float(np.real(np.trace(rho @ sz)))
        return np.array([x, y, z])

    # ------------------------------------------------------------------
    # Fidelity & distance
    # ------------------------------------------------------------------

    def fidelity(self, other: "QuantumState") -> float:
        """State fidelity |⟨ψ|φ⟩|²."""
        return float(abs(np.vdot(self._sv, other._sv)) ** 2)

    def trace_distance(self, other: "QuantumState") -> float:
        """Trace distance ½ ‖ρ - σ‖₁."""
        diff = self.density_matrix() - other.density_matrix()
        eigenvalues = np.linalg.eigvalsh(diff)
        return float(0.5 * np.sum(np.abs(eigenvalues)))

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def copy(self) -> "QuantumState":
        """Deep copy."""
        qs = QuantumState(self.n_qubits, seed=None)
        qs._sv = self._sv.copy()
        qs._rng = np.random.default_rng()
        return qs

    def __repr__(self) -> str:
        return (
            f"QuantumState(n_qubits={self.n_qubits}, "
            f"norm={self.norm:.6f})"
        )
