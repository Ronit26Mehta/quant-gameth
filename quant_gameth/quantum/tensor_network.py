"""
Matrix Product State (MPS) tensor network backend for large qubit counts.

From Tech Spec §2.2.1 and Math Foundations §F.2:
    |ψ⟩ = Σ A[1]^{s₁} A[2]^{s₂} … A[n]^{sₙ} |s₁…sₙ⟩
    where A[i]^{sᵢ} are χ×χ matrices (bond dimension χ).

Memory: O(nχ²)  vs  O(2^n) for full statevector.
Gate application: local tensor update + SVD truncation.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional, Tuple


class MatrixProductState:
    """MPS representation of an n-qubit quantum state.

    Parameters
    ----------
    n_qubits : int
    max_bond_dim : int
        Maximum bond dimension χ.  Higher = more accurate but slower.
    """

    def __init__(self, n_qubits: int, max_bond_dim: int = 64):
        self.n_qubits = n_qubits
        self.max_bond_dim = max_bond_dim
        # Initialise to |00…0⟩
        self.tensors: List[np.ndarray] = []
        for i in range(n_qubits):
            # Each tensor has shape (bond_left, physical=2, bond_right)
            t = np.zeros((1, 2, 1), dtype=np.complex128)
            t[0, 0, 0] = 1.0  # |0⟩
            self.tensors.append(t)

    @classmethod
    def uniform_superposition(cls, n_qubits: int, max_bond_dim: int = 64) -> "MatrixProductState":
        """Create |+⟩^⊗n as MPS."""
        mps = cls(n_qubits, max_bond_dim)
        for i in range(n_qubits):
            t = np.zeros((1, 2, 1), dtype=np.complex128)
            t[0, 0, 0] = 1.0 / np.sqrt(2)
            t[0, 1, 0] = 1.0 / np.sqrt(2)
            mps.tensors[i] = t
        return mps

    def apply_single_qubit_gate(self, gate: np.ndarray, qubit: int) -> None:
        """Apply a 2×2 gate to a single qubit.

        Simply contracts the gate with the physical index.
        No SVD needed — bond dimensions unchanged.
        """
        # tensors[qubit] shape: (χ_left, 2, χ_right)
        # gate shape: (2, 2)
        # result shape: (χ_left, 2, χ_right)
        t = self.tensors[qubit]
        # Contract: new_t[l, s', r] = Σ_s gate[s', s] * t[l, s, r]
        new_t = np.einsum("ij,ljr->lir", gate, t)
        self.tensors[qubit] = new_t

    def apply_two_qubit_gate(
        self, gate: np.ndarray, qubit1: int, qubit2: int
    ) -> None:
        """Apply a 4×4 gate to adjacent qubits.

        Uses SVD truncation to maintain bond dimension.
        For non-adjacent qubits, swaps are applied first.
        """
        if abs(qubit1 - qubit2) != 1:
            # Need to bring qubits adjacent via SWAP
            self._swap_to_adjacent(qubit1, qubit2)
            q_min = min(qubit1, qubit2)
            self._apply_adjacent_gate(gate, q_min)
            self._swap_to_adjacent(qubit1, qubit2)  # swap back
        else:
            q_min = min(qubit1, qubit2)
            if qubit1 > qubit2:
                # Reorder gate matrix for swapped qubit order
                swap = np.array([
                    [1, 0, 0, 0],
                    [0, 0, 1, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1],
                ], dtype=np.complex128)
                gate = swap @ gate @ swap
            self._apply_adjacent_gate(gate, q_min)

    def _apply_adjacent_gate(self, gate: np.ndarray, q: int) -> None:
        """Apply gate to qubits q and q+1 (must be adjacent).

        Following §F.2:
        1. Contract A[q] and A[q+1] into a single tensor
        2. Apply gate
        3. SVD + truncation
        4. Split back into A[q] and A[q+1]
        """
        t1 = self.tensors[q]      # shape: (χ_l, 2, χ_m)
        t2 = self.tensors[q + 1]  # shape: (χ_m, 2, χ_r)

        χ_l = t1.shape[0]
        χ_r = t2.shape[2]

        # Contract into (χ_l, 2, 2, χ_r)
        theta = np.einsum("lsr,rmR->lsmR", t1, t2)  # (χ_l, 2, 2, χ_r)

        # Apply gate: gate is 4×4, reshape to (2,2,2,2)
        gate_4d = gate.reshape(2, 2, 2, 2)
        # theta_new[l, s1', s2', r] = Σ_{s1,s2} gate[s1', s2', s1, s2] * theta[l, s1, s2, r]
        theta_new = np.einsum("abij,lijR->labR", gate_4d, theta)

        # Reshape for SVD: (χ_l * 2, 2 * χ_r)
        theta_mat = theta_new.reshape(χ_l * 2, 2 * χ_r)

        # SVD with truncation
        U, S, Vh = np.linalg.svd(theta_mat, full_matrices=False)

        # Truncate to max_bond_dim
        χ = min(len(S), self.max_bond_dim)
        U = U[:, :χ]
        S = S[:χ]
        Vh = Vh[:χ, :]

        # Absorb singular values symmetrically
        sqrt_S = np.sqrt(S)
        U_scaled = U * sqrt_S[np.newaxis, :]
        Vh_scaled = sqrt_S[:, np.newaxis] * Vh

        # Reshape back to tensors
        self.tensors[q] = U_scaled.reshape(χ_l, 2, χ)
        self.tensors[q + 1] = Vh_scaled.reshape(χ, 2, χ_r)

    def _swap_to_adjacent(self, q1: int, q2: int) -> None:
        """Bring qubits q1, q2 adjacent via SWAP gates."""
        swap_gate = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ], dtype=np.complex128)
        if q1 > q2:
            q1, q2 = q2, q1
        for q in range(q1, q2 - 1):
            self._apply_adjacent_gate(swap_gate, q)

    def get_amplitude(self, bitstring: str) -> complex:
        """Get amplitude ⟨bitstring|ψ⟩ by contracting the MPS."""
        result = np.array([[1.0]], dtype=np.complex128)
        for i, bit in enumerate(bitstring):
            b = int(bit)
            # Select physical index
            mat = self.tensors[i][:, b, :]  # shape: (χ_l, χ_r)
            result = result @ mat
        return complex(result[0, 0])

    def get_probabilities(self, max_states: int = 4096) -> dict:
        """Compute probabilities for all basis states (up to max_states).

        For large n, only returns the most significant states.
        """
        n = self.n_qubits
        dim = min(1 << n, max_states)
        probs = {}
        for x in range(dim):
            bs = format(x, f"0{n}b")
            amp = self.get_amplitude(bs)
            prob = abs(amp) ** 2
            if prob > 1e-12:
                probs[bs] = float(prob)
        return probs

    def norm(self) -> float:
        """Compute ⟨ψ|ψ⟩ via MPS contraction."""
        # Contract from left
        env = np.eye(1, dtype=np.complex128)
        for t in self.tensors:
            # env shape: (χ, χ)
            # t shape: (χ_l, 2, χ_r)
            # contract: new_env[r, r'] = Σ_{l,s} env[l, l'] * t[l, s, r] * conj(t[l', s, r'])
            env = np.einsum("ll,lsr,LSR->rR", env, t, np.conj(t))
        return float(np.real(env[0, 0]))

    def to_statevector(self) -> np.ndarray:
        """Convert MPS back to full statevector (only for small n)."""
        n = self.n_qubits
        if n > 20:
            raise ValueError("Statevector too large for n > 20 qubits")
        dim = 1 << n
        sv = np.zeros(dim, dtype=np.complex128)
        for x in range(dim):
            bs = format(x, f"0{n}b")
            sv[x] = self.get_amplitude(bs)
        return sv

    def entanglement_entropy(self, cut_position: int) -> float:
        """Von Neumann entropy of the bipartition at cut_position.

        Computed from singular values at the bond.
        """
        # Contract tensors up to cut_position
        env = np.eye(self.tensors[0].shape[0], dtype=np.complex128)
        for i in range(cut_position):
            t = self.tensors[i]
            # env shape: (χ, χ')
            # t shape: (χ_l, 2, χ_r)
            env = np.einsum("ll,lsr,LSR->rR", env, t, np.conj(t))

        # Eigenvalues of reduced density matrix
        eigvals = np.linalg.eigvalsh(env)
        eigvals = eigvals[eigvals > 1e-15]
        total = eigvals.sum()
        if total > 0:
            eigvals = eigvals / total
        return float(-np.sum(eigvals * np.log2(eigvals + 1e-30)))
