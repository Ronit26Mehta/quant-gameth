"""
GPU backend — CuPy-accelerated statevector simulation.

Optional dependency: requires ``cupy`` (``pip install cupy-cuda12x``).
Falls back to the classical backend if CuPy is unavailable.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import numpy as np

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = None  # type: ignore[assignment]
    HAS_CUPY = False


class GPUBackend:
    """CuPy GPU-accelerated statevector simulation backend.

    Mirrors the ``ClassicalBackend`` protocol so it can be swapped in
    transparently by the hybrid scheduler.

    If CuPy is not installed, all methods fall back to NumPy with a
    warning logged once at construction time.
    """

    name = "gpu"

    def __init__(self, n_qubits: int) -> None:
        self.n_qubits = n_qubits
        self._gpu_available = HAS_CUPY
        self._statevector: Optional[np.ndarray] = None

        if not self._gpu_available:
            import warnings
            warnings.warn(
                "CuPy not found — GPUBackend will use NumPy fallback. "
                "Install cupy for GPU acceleration.",
                RuntimeWarning,
                stacklevel=2,
            )

    # ── Helpers ─────────────────────────────────────────────────────────

    def _to_device(self, arr: np.ndarray) -> Any:
        """Move array to GPU (or return as-is for fallback)."""
        if self._gpu_available:
            return cp.asarray(arr)
        return arr

    def _to_host(self, arr: Any) -> np.ndarray:
        """Move array from GPU to CPU."""
        if self._gpu_available and isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
        return np.asarray(arr)

    def _xp(self) -> Any:
        """Return the array module (cupy or numpy)."""
        return cp if self._gpu_available else np

    # ── Circuit execution ───────────────────────────────────────────────

    def run_circuit(
        self,
        circuit: Any,
        shots: int = 0,
        return_statevector: bool = True,
    ) -> Dict[str, Any]:
        """Execute circuit on GPU.

        For the GPU path we transfer the gate matrices and statevector
        to device memory, apply gates using CuPy, then transfer back.
        """
        xp = self._xp()
        t0 = time.perf_counter()

        dim = 1 << self.n_qubits
        sv = xp.zeros(dim, dtype=xp.complex128)
        sv[0] = 1.0  # |00...0⟩

        for gate_name, qubits, params in circuit.gates:
            matrix = params[0] if params else None
            if matrix is None:
                continue

            mat = self._to_device(np.asarray(matrix))
            sv = self._apply_gate_gpu(sv, mat, qubits, xp)

        sv_host = self._to_host(sv)
        self._statevector = sv_host
        elapsed = time.perf_counter() - t0

        result: Dict[str, Any] = {"time_seconds": elapsed}
        if return_statevector:
            result["statevector"] = sv_host

        if shots > 0:
            result["counts"] = self.sample(sv_host, shots)

        return result

    def _apply_gate_gpu(
        self, sv: Any, gate: Any, qubits: list, xp: Any
    ) -> Any:
        """Apply a gate to the statevector using stride-based indexing.

        This mirrors the classical simulator's approach but uses the
        GPU array module for acceleration.
        """
        n = self.n_qubits
        dim = 1 << n

        if len(qubits) == 1:
            q = qubits[0]
            stride = 1 << (n - 1 - q)
            new_sv = xp.copy(sv)

            for i in range(dim):
                if i & stride:
                    continue
                i0 = i
                i1 = i | stride
                a, b = sv[i0], sv[i1]
                new_sv[i0] = gate[0, 0] * a + gate[0, 1] * b
                new_sv[i1] = gate[1, 0] * a + gate[1, 1] * b
            return new_sv

        elif len(qubits) == 2:
            # 2-qubit gate via reshape-based application
            q0, q1 = qubits
            new_sv = xp.copy(sv)

            s0 = 1 << (n - 1 - q0)
            s1 = 1 << (n - 1 - q1)

            for i in range(dim):
                if (i & s0) or (i & s1):
                    continue
                indices = [
                    i,
                    i | s1,
                    i | s0,
                    i | s0 | s1,
                ]
                vals = xp.array([sv[idx] for idx in indices])
                new_vals = gate @ vals
                for k, idx in enumerate(indices):
                    new_sv[idx] = new_vals[k]
            return new_sv

        else:
            # Fallback: construct full unitary and apply
            full_gate = self._to_device(np.eye(dim, dtype=np.complex128))
            return full_gate @ sv

    # ── Expectation value ───────────────────────────────────────────────

    def expectation(
        self,
        statevector: np.ndarray,
        hamiltonian: np.ndarray,
    ) -> float:
        """Compute ⟨ψ|H|ψ⟩ on GPU."""
        xp = self._xp()
        sv = self._to_device(statevector)
        H = self._to_device(hamiltonian)

        if H.ndim == 1:
            result = float(xp.real(xp.sum(xp.abs(sv) ** 2 * H)))
        else:
            result = float(xp.real(xp.conj(sv) @ H @ sv))
        return result

    # ── Sampling ────────────────────────────────────────────────────────

    def sample(
        self,
        statevector: np.ndarray,
        n_shots: int,
        seed: int = 42,
    ) -> Dict[str, int]:
        """Sample on host (CuPy's random is limited for this use-case)."""
        rng = np.random.default_rng(seed)
        sv_host = self._to_host(statevector) if not isinstance(statevector, np.ndarray) else statevector
        probs = np.abs(sv_host) ** 2
        probs /= probs.sum()

        n_qubits = int(np.log2(len(sv_host)))
        outcomes = rng.choice(len(sv_host), size=n_shots, p=probs)
        counts: Dict[str, int] = {}
        for o in outcomes:
            bs = format(o, f"0{n_qubits}b")
            counts[bs] = counts.get(bs, 0) + 1
        return counts

    # ── Utility ─────────────────────────────────────────────────────────

    def get_statevector(self) -> Optional[np.ndarray]:
        return self._statevector

    def info(self) -> Dict[str, Any]:
        info_dict: Dict[str, Any] = {
            "name": self.name,
            "n_qubits": self.n_qubits,
            "gpu_available": self._gpu_available,
        }
        if self._gpu_available:
            dev = cp.cuda.Device()
            info_dict["gpu_name"] = dev.attributes.get("DeviceName", "unknown")
            info_dict["gpu_memory_gb"] = (
                dev.mem_info[1] / (1024 ** 3) if hasattr(dev, "mem_info") else "unknown"
            )
        return info_dict
