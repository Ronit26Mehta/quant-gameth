"""
Classical backend — NumPy/SciPy statevector simulation engine.

This is the default execution backend for circuits ≤ ~20 qubits.
It wraps the core statevector simulator with a uniform Backend protocol
so dispatchers can swap in GPU/tensor-network alternatives transparently.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from quant_gameth._types import SolverResult, SolverMethod


class ClassicalBackend:
    """NumPy-based statevector simulation backend.

    Protocol
    --------
    Every backend exposes the same three entry points:

    * ``run_circuit(circuit, shots)``  — execute a circuit
    * ``expectation(circuit, hamiltonian)`` — compute ⟨ψ|H|ψ⟩
    * ``sample(circuit, n_shots)`` — sample measurement outcomes
    """

    name = "classical"

    def __init__(self, n_qubits: int, *, dtype: type = np.complex128) -> None:
        self.n_qubits = n_qubits
        self.dtype = dtype
        self._statevector: Optional[np.ndarray] = None

    # ── Circuit execution ───────────────────────────────────────────────

    def run_circuit(
        self,
        circuit: Any,
        shots: int = 0,
        return_statevector: bool = True,
    ) -> Dict[str, Any]:
        """Simulate a circuit and optionally sample.

        Parameters
        ----------
        circuit : QuantumCircuit
            Circuit to execute.
        shots : int
            If > 0, sample measurement outcomes.
        return_statevector : bool
            If True, include the final statevector in the result.

        Returns
        -------
        dict with keys: ``statevector``, ``counts``, ``time_seconds``.
        """
        from quant_gameth.quantum.circuit import Simulator

        t0 = time.perf_counter()
        sim = Simulator()
        sv = sim.run(circuit)
        self._statevector = sv
        elapsed = time.perf_counter() - t0

        result: Dict[str, Any] = {"time_seconds": elapsed}

        if return_statevector:
            result["statevector"] = sv

        if shots > 0:
            probs = np.abs(sv) ** 2
            probs /= probs.sum()
            outcomes = np.random.default_rng().choice(len(sv), size=shots, p=probs)
            counts: Dict[str, int] = {}
            for o in outcomes:
                bs = format(o, f"0{self.n_qubits}b")
                counts[bs] = counts.get(bs, 0) + 1
            result["counts"] = counts

        return result

    # ── Expectation value ───────────────────────────────────────────────

    def expectation(
        self,
        statevector: np.ndarray,
        hamiltonian: np.ndarray,
    ) -> float:
        """Compute ⟨ψ|H|ψ⟩.

        Parameters
        ----------
        statevector : np.ndarray, shape ``(2**n,)``
        hamiltonian : np.ndarray
            Either diagonal (shape ``(2**n,)``) or dense (shape ``(2**n, 2**n)``).
        """
        if hamiltonian.ndim == 1:
            # Diagonal Hamiltonian — fast path
            return float(np.real(np.sum(np.abs(statevector) ** 2 * hamiltonian)))
        else:
            return float(np.real(np.conj(statevector) @ hamiltonian @ statevector))

    # ── Sampling ────────────────────────────────────────────────────────

    def sample(
        self,
        statevector: np.ndarray,
        n_shots: int,
        seed: int = 42,
    ) -> Dict[str, int]:
        """Sample computational basis measurements.

        Returns
        -------
        dict mapping bitstring → count.
        """
        rng = np.random.default_rng(seed)
        probs = np.abs(statevector) ** 2
        probs /= probs.sum()

        n_qubits = int(np.log2(len(statevector)))
        outcomes = rng.choice(len(statevector), size=n_shots, p=probs)

        counts: Dict[str, int] = {}
        for o in outcomes:
            bs = format(o, f"0{n_qubits}b")
            counts[bs] = counts.get(bs, 0) + 1
        return counts

    # ── Utility ─────────────────────────────────────────────────────────

    def get_statevector(self) -> Optional[np.ndarray]:
        """Return the last-computed statevector."""
        return self._statevector

    def probabilities(self, statevector: np.ndarray) -> np.ndarray:
        """Born-rule probabilities from a statevector."""
        return np.abs(statevector) ** 2

    def info(self) -> Dict[str, Any]:
        """Return backend metadata."""
        return {
            "name": self.name,
            "n_qubits": self.n_qubits,
            "dtype": str(self.dtype),
            "max_memory_gb": (2 ** self.n_qubits * np.dtype(self.dtype).itemsize)
                             / (1024 ** 3),
        }
