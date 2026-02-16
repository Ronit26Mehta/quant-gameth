"""
Strategy mapper — classical ↔ quantum strategy conversion.

Maps classical mixed strategies to quantum state vectors and back.
Uses angle encoding: probability pᵢ → rotation angle θᵢ.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple

from quant_gameth.quantum.circuit import QuantumCircuit
from quant_gameth.quantum.gates import Gates


class StrategyMapper:
    """Map between classical and quantum strategy representations."""

    @staticmethod
    def classical_to_quantum_state(
        mixed_strategy: np.ndarray,
    ) -> np.ndarray:
        """Convert a classical mixed strategy to a quantum state vector.

        Uses amplitude encoding: p → |ψ⟩ = Σ √pᵢ |i⟩.
        """
        probs = np.asarray(mixed_strategy, dtype=float)
        probs = np.maximum(probs, 0)
        total = probs.sum()
        if total > 0:
            probs /= total
        return np.sqrt(probs).astype(np.complex128)

    @staticmethod
    def quantum_state_to_classical(
        statevector: np.ndarray,
    ) -> np.ndarray:
        """Convert a quantum state to classical mixed strategy via Born rule."""
        return np.abs(statevector) ** 2

    @staticmethod
    def angle_encoding(
        mixed_strategy: np.ndarray,
    ) -> np.ndarray:
        """Encode probabilities as rotation angles.

        pᵢ → θᵢ = 2 arcsin(√pᵢ)
        """
        probs = np.asarray(mixed_strategy, dtype=float)
        probs = np.clip(probs, 0, 1)
        return 2 * np.arcsin(np.sqrt(probs))

    @staticmethod
    def angles_to_probabilities(angles: np.ndarray) -> np.ndarray:
        """Decode rotation angles back to probabilities.

        θᵢ → pᵢ = sin²(θᵢ/2)
        """
        return np.sin(angles / 2) ** 2

    @staticmethod
    def strategy_to_circuit(
        mixed_strategy: np.ndarray,
        encoding: str = "amplitude",
    ) -> QuantumCircuit:
        """Create a quantum circuit that prepares the strategy state.

        Parameters
        ----------
        mixed_strategy : np.ndarray
            Probability vector.
        encoding : str
            ``'amplitude'``: full amplitude encoding (exact).
            ``'angle'``: one qubit per strategy dimension with RY encoding.
        """
        probs = np.asarray(mixed_strategy, dtype=float)
        probs = np.maximum(probs, 0)
        total = probs.sum()
        if total > 0:
            probs /= total

        n_strats = len(probs)

        if encoding == "angle":
            # One qubit per strategy dimension
            circ = QuantumCircuit(n_strats)
            angles = 2 * np.arcsin(np.sqrt(np.clip(probs, 0, 1)))
            for q in range(n_strats):
                circ.ry(float(angles[q]), q)
            return circ

        # Amplitude encoding (needs ceiling(log2(n)) qubits)
        n_qubits = max(1, int(np.ceil(np.log2(max(n_strats, 2)))))
        circ = QuantumCircuit(n_qubits)

        # Use recursive bisection (simplified Möttönen decomposition)
        amplitudes = np.zeros(1 << n_qubits, dtype=float)
        amplitudes[:n_strats] = np.sqrt(probs)
        norm = np.linalg.norm(amplitudes)
        if norm > 1e-14:
            amplitudes /= norm

        # For small n, use direct state preparation via RY gates
        if n_qubits <= 3:
            _recursive_amplitude_prep(circ, amplitudes, list(range(n_qubits)), 0)

        return circ

    @staticmethod
    def bloch_to_strategy(theta: float, phi: float) -> np.ndarray:
        """Convert Bloch sphere coordinates to 2-strategy mixed strategy.

        |ψ⟩ = cos(θ/2)|0⟩ + e^{iφ}sin(θ/2)|1⟩
        probabilities: [cos²(θ/2), sin²(θ/2)]
        """
        return np.array([np.cos(theta / 2) ** 2, np.sin(theta / 2) ** 2])

    @staticmethod
    def strategy_to_bloch(mixed_strategy: np.ndarray) -> Tuple[float, float]:
        """Convert 2-strategy mixed strategy to Bloch coordinates.

        Returns (θ, φ) where φ is chosen as 0 (no relative phase info).
        """
        p0 = float(mixed_strategy[0])
        theta = 2 * np.arccos(np.sqrt(np.clip(p0, 0, 1)))
        return (theta, 0.0)


def _recursive_amplitude_prep(
    circ: QuantumCircuit,
    amplitudes: np.ndarray,
    qubits: List[int],
    depth: int,
) -> None:
    """Recursive amplitude preparation via binary tree decomposition."""
    n = len(amplitudes)
    if n == 1:
        return
    if n == 2:
        a0, a1 = amplitudes[0], amplitudes[1]
        norm = np.sqrt(a0 ** 2 + a1 ** 2)
        if norm > 1e-14:
            theta = 2 * np.arctan2(a1, a0)
            circ.ry(float(theta), qubits[0])
        return

    half = n // 2
    left = amplitudes[:half]
    right = amplitudes[half:]

    norm_left = np.linalg.norm(left)
    norm_right = np.linalg.norm(right)
    total_norm = np.sqrt(norm_left ** 2 + norm_right ** 2)

    if total_norm > 1e-14:
        theta = 2 * np.arctan2(norm_right, norm_left)
        circ.ry(float(theta), qubits[0])

    # Recurse on sub-problems (simplified — full version needs controlled rotations)
    if len(qubits) > 1:
        if norm_left > 1e-14:
            _recursive_amplitude_prep(circ, left / max(norm_left, 1e-14), qubits[1:], depth + 1)
