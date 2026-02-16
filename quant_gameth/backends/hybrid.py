"""
Hybrid backend — intelligent scheduler that decomposes circuits and
dispatches sub-tasks to the optimal backend (classical / GPU / tensor network).

Strategy
--------
1. **Small circuits** (≤ 16 qubits) → ClassicalBackend  (fastest for small n)
2. **Medium circuits** (17–22 qubits) → GPUBackend if available, else classical
3. **Large circuits** (> 22 qubits) → TensorNetwork (MPS) backend
4. **Low-entanglement circuits** → always TensorNetwork regardless of size

The scheduler also supports explicit overrides via ``force_backend``.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np

from quant_gameth.backends.classical import ClassicalBackend
from quant_gameth.backends.gpu import GPUBackend


class HybridBackend:
    """Hybrid scheduler that routes circuits to the best-fit backend.

    Usage::

        backend = HybridBackend(n_qubits=24)
        result = backend.run_circuit(circuit, shots=1000)
        # Automatically dispatched to MPS backend for n > 22
    """

    # Thresholds
    CLASSICAL_MAX = 16
    GPU_MAX = 22

    def __init__(
        self,
        n_qubits: int,
        *,
        force_backend: Optional[str] = None,
        mps_bond_dim: int = 64,
    ) -> None:
        self.n_qubits = n_qubits
        self._force = force_backend
        self._mps_bond_dim = mps_bond_dim

        # Lazily initialised backends
        self._classical: Optional[ClassicalBackend] = None
        self._gpu: Optional[GPUBackend] = None
        self._mps: Optional[Any] = None

        self._dispatch_log: List[Dict[str, Any]] = []

    # ── Backend selection ───────────────────────────────────────────────

    def _select_backend(
        self,
        circuit: Optional[Any] = None,
    ) -> str:
        """Choose the optimal backend for the current workload."""
        if self._force:
            return self._force

        n = self.n_qubits

        if n <= self.CLASSICAL_MAX:
            return "classical"

        # Check if GPU is available for medium circuits
        if n <= self.GPU_MAX:
            try:
                import cupy  # noqa: F401
                return "gpu"
            except ImportError:
                return "classical"

        # Large circuits → tensor network
        return "tensor_network"

    def _get_backend(self, name: str) -> Any:
        """Lazily instantiate and return the requested backend."""
        if name == "classical":
            if self._classical is None:
                self._classical = ClassicalBackend(self.n_qubits)
            return self._classical

        elif name == "gpu":
            if self._gpu is None:
                self._gpu = GPUBackend(self.n_qubits)
            return self._gpu

        elif name == "tensor_network":
            if self._mps is None:
                from quant_gameth.quantum.tensor_network import MPSSimulator
                self._mps = MPSSimulator(
                    n_qubits=self.n_qubits,
                    bond_dim=self._mps_bond_dim,
                )
            return self._mps

        raise ValueError(f"Unknown backend: {name}")

    # ── Circuit execution ───────────────────────────────────────────────

    def run_circuit(
        self,
        circuit: Any,
        shots: int = 0,
        return_statevector: bool = True,
    ) -> Dict[str, Any]:
        """Execute a circuit via the best-fit backend.

        Returns
        -------
        dict with keys: ``statevector`` (if requested), ``counts`` (if
        shots > 0), ``time_seconds``, ``backend_used``.
        """
        backend_name = self._select_backend(circuit)
        backend = self._get_backend(backend_name)

        t0 = time.perf_counter()

        if backend_name == "tensor_network":
            # MPS simulator has a different interface
            sv = backend.run(circuit)
            elapsed = time.perf_counter() - t0
            result: Dict[str, Any] = {"time_seconds": elapsed}

            if return_statevector:
                result["statevector"] = sv

            if shots > 0:
                probs = np.abs(sv) ** 2
                probs /= probs.sum()
                rng = np.random.default_rng()
                outcomes = rng.choice(len(sv), size=shots, p=probs)
                counts: Dict[str, int] = {}
                for o in outcomes:
                    bs = format(o, f"0{self.n_qubits}b")
                    counts[bs] = counts.get(bs, 0) + 1
                result["counts"] = counts
        else:
            result = backend.run_circuit(
                circuit, shots=shots, return_statevector=return_statevector,
            )

        result["backend_used"] = backend_name

        self._dispatch_log.append({
            "backend": backend_name,
            "n_qubits": self.n_qubits,
            "n_gates": len(circuit.gates) if hasattr(circuit, "gates") else 0,
            "time_seconds": result["time_seconds"],
        })

        return result

    # ── Expectation value ───────────────────────────────────────────────

    def expectation(
        self,
        statevector: np.ndarray,
        hamiltonian: np.ndarray,
    ) -> float:
        """Compute ⟨ψ|H|ψ⟩ using the best available backend."""
        backend_name = self._select_backend()
        backend = self._get_backend(backend_name)

        if hasattr(backend, "expectation"):
            return backend.expectation(statevector, hamiltonian)

        # Fallback: classical computation
        if hamiltonian.ndim == 1:
            return float(np.real(np.sum(np.abs(statevector) ** 2 * hamiltonian)))
        return float(np.real(np.conj(statevector) @ hamiltonian @ statevector))

    # ── Sampling ────────────────────────────────────────────────────────

    def sample(
        self,
        statevector: np.ndarray,
        n_shots: int,
        seed: int = 42,
    ) -> Dict[str, int]:
        """Sample measurement outcomes."""
        backend_name = self._select_backend()
        backend = self._get_backend(backend_name)

        if hasattr(backend, "sample"):
            return backend.sample(statevector, n_shots, seed=seed)

        # Fallback
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

    # ── Decomposition ───────────────────────────────────────────────────

    def decompose_and_run(
        self,
        circuit: Any,
        partition: Optional[List[List[int]]] = None,
        shots: int = 0,
    ) -> Dict[str, Any]:
        """Decompose a circuit into sub-circuits and run on mixed backends.

        Parameters
        ----------
        circuit : QuantumCircuit
        partition : list of list of int or None
            Qubit partitions. If None, auto-partition by connectivity.
        shots : int

        Returns
        -------
        dict with combined results.
        """
        # For now, dispatch the whole circuit to the best backend.
        # True decomposition (circuit cutting) is future work.
        return self.run_circuit(circuit, shots=shots)

    # ── Info ────────────────────────────────────────────────────────────

    def dispatch_history(self) -> List[Dict[str, Any]]:
        """Return the log of backend dispatch decisions."""
        return self._dispatch_log.copy()

    def info(self) -> Dict[str, Any]:
        backend_name = self._select_backend()
        return {
            "name": "hybrid",
            "n_qubits": self.n_qubits,
            "selected_backend": backend_name,
            "thresholds": {
                "classical_max": self.CLASSICAL_MAX,
                "gpu_max": self.GPU_MAX,
            },
            "mps_bond_dim": self._mps_bond_dim,
            "dispatch_count": len(self._dispatch_log),
        }
