"""
Quantum circuit builder and statevector simulator.

Follows the implementation strategy from Mathematical Foundations §F.1:
    - Single-qubit gates: stride-based index arithmetic O(2^n) per gate
    - Two-qubit gates: 4-amplitude block update O(2^n) per gate
    - Measurement: Born-rule probability, shot-based sampling
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from quant_gameth.quantum.gates import Gates


# ---------------------------------------------------------------------------
# Gate instruction (stored in circuit)
# ---------------------------------------------------------------------------

class GateInstruction:
    """A single gate application in a circuit."""
    __slots__ = ("gate", "qubits", "label")

    def __init__(
        self, gate: np.ndarray, qubits: Tuple[int, ...], label: str = ""
    ):
        self.gate = gate
        self.qubits = qubits
        self.label = label

    def __repr__(self) -> str:
        return f"Gate({self.label}, qubits={self.qubits})"


# ---------------------------------------------------------------------------
# QuantumCircuit
# ---------------------------------------------------------------------------

class QuantumCircuit:
    """Construct a quantum circuit by appending gate instructions.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the register.
    """

    def __init__(self, n_qubits: int):
        if n_qubits < 1:
            raise ValueError("n_qubits must be >= 1")
        self.n_qubits: int = n_qubits
        self._instructions: List[GateInstruction] = []

    # ----- convenience gate methods -----

    def h(self, q: int) -> "QuantumCircuit":
        self._append_1q(Gates.H(), q, "H")
        return self

    def x(self, q: int) -> "QuantumCircuit":
        self._append_1q(Gates.X(), q, "X")
        return self

    def y(self, q: int) -> "QuantumCircuit":
        self._append_1q(Gates.Y(), q, "Y")
        return self

    def z(self, q: int) -> "QuantumCircuit":
        self._append_1q(Gates.Z(), q, "Z")
        return self

    def s(self, q: int) -> "QuantumCircuit":
        self._append_1q(Gates.S(), q, "S")
        return self

    def t(self, q: int) -> "QuantumCircuit":
        self._append_1q(Gates.T(), q, "T")
        return self

    def rx(self, theta: float, q: int) -> "QuantumCircuit":
        self._append_1q(Gates.Rx(theta), q, f"Rx({theta:.4f})")
        return self

    def ry(self, theta: float, q: int) -> "QuantumCircuit":
        self._append_1q(Gates.Ry(theta), q, f"Ry({theta:.4f})")
        return self

    def rz(self, theta: float, q: int) -> "QuantumCircuit":
        self._append_1q(Gates.Rz(theta), q, f"Rz({theta:.4f})")
        return self

    def u3(self, theta: float, phi: float, lam: float, q: int) -> "QuantumCircuit":
        self._append_1q(Gates.U3(theta, phi, lam), q, "U3")
        return self

    def phase(self, phi: float, q: int) -> "QuantumCircuit":
        self._append_1q(Gates.Phase(phi), q, f"P({phi:.4f})")
        return self

    def cx(self, control: int, target: int) -> "QuantumCircuit":
        self._append_2q(Gates.CNOT(), control, target, "CX")
        return self

    def cnot(self, control: int, target: int) -> "QuantumCircuit":
        return self.cx(control, target)

    def cz(self, q1: int, q2: int) -> "QuantumCircuit":
        self._append_2q(Gates.CZ(), q1, q2, "CZ")
        return self

    def swap(self, q1: int, q2: int) -> "QuantumCircuit":
        self._append_2q(Gates.SWAP(), q1, q2, "SWAP")
        return self

    def crx(self, theta: float, control: int, target: int) -> "QuantumCircuit":
        self._append_2q(Gates.CRx(theta), control, target, "CRx")
        return self

    def cry(self, theta: float, control: int, target: int) -> "QuantumCircuit":
        self._append_2q(Gates.CRy(theta), control, target, "CRy")
        return self

    def crz(self, theta: float, control: int, target: int) -> "QuantumCircuit":
        self._append_2q(Gates.CRz(theta), control, target, "CRz")
        return self

    def rzz(self, theta: float, q1: int, q2: int) -> "QuantumCircuit":
        self._append_2q(Gates.RZZ(theta), q1, q2, "RZZ")
        return self

    def rxx(self, theta: float, q1: int, q2: int) -> "QuantumCircuit":
        self._append_2q(Gates.RXX(theta), q1, q2, "RXX")
        return self

    def toffoli(self, c1: int, c2: int, target: int) -> "QuantumCircuit":
        self._instructions.append(GateInstruction(
            Gates.Toffoli(), (c1, c2, target), "CCX"
        ))
        return self

    def custom(
        self, gate: np.ndarray, qubits: Tuple[int, ...], label: str = "U"
    ) -> "QuantumCircuit":
        """Append an arbitrary gate matrix on specified qubits."""
        self._instructions.append(GateInstruction(gate, qubits, label))
        return self

    # ----- circuit metadata -----

    @property
    def depth(self) -> int:
        """Circuit depth (longest chain of sequential gates on any qubit)."""
        if not self._instructions:
            return 0
        layers: Dict[int, int] = {}
        for inst in self._instructions:
            max_layer = max((layers.get(q, 0) for q in inst.qubits), default=0)
            for q in inst.qubits:
                layers[q] = max_layer + 1
        return max(layers.values())

    @property
    def size(self) -> int:
        """Total number of gate operations."""
        return len(self._instructions)

    @property
    def instructions(self) -> List[GateInstruction]:
        return list(self._instructions)

    # ----- internal helpers -----

    def _append_1q(self, gate: np.ndarray, q: int, label: str) -> None:
        self._validate_qubit(q)
        self._instructions.append(GateInstruction(gate, (q,), label))

    def _append_2q(
        self, gate: np.ndarray, q1: int, q2: int, label: str
    ) -> None:
        self._validate_qubit(q1)
        self._validate_qubit(q2)
        if q1 == q2:
            raise ValueError("Two-qubit gate requires distinct qubits")
        self._instructions.append(GateInstruction(gate, (q1, q2), label))

    def _validate_qubit(self, q: int) -> None:
        if not (0 <= q < self.n_qubits):
            raise ValueError(
                f"Qubit index {q} out of range [0, {self.n_qubits})"
            )

    def __repr__(self) -> str:
        return (
            f"QuantumCircuit(n_qubits={self.n_qubits}, "
            f"depth={self.depth}, size={self.size})"
        )


# ---------------------------------------------------------------------------
# Simulator (statevector)
# ---------------------------------------------------------------------------

class Simulator:
    """Execute a QuantumCircuit via statevector simulation.

    Follows the stride-based gate application from Math Foundations §F.1
    for maximum efficiency without full tensor-product expansion.
    """

    @staticmethod
    def run(
        circuit: QuantumCircuit,
        initial_state: Optional[np.ndarray] = None,
        seed: Optional[int] = 42,
    ) -> SimulatorResult:
        """Execute a circuit and return the final state + timing.

        Parameters
        ----------
        circuit : QuantumCircuit
        initial_state : np.ndarray or None
            If ``None``, starts from |00…0⟩.
        seed : int or None
            Seed for measurement sampling.

        Returns
        -------
        SimulatorResult
        """
        n = circuit.n_qubits
        dim = 1 << n
        if initial_state is not None:
            sv = np.array(initial_state, dtype=np.complex128).copy()
        else:
            sv = np.zeros(dim, dtype=np.complex128)
            sv[0] = 1.0

        t0 = time.perf_counter()

        for inst in circuit.instructions:
            nq = len(inst.qubits)
            if nq == 1:
                sv = _apply_single_qubit_gate(sv, inst.gate, inst.qubits[0], n)
            elif nq == 2:
                sv = _apply_two_qubit_gate(
                    sv, inst.gate, inst.qubits[0], inst.qubits[1], n
                )
            elif nq == 3:
                sv = _apply_multi_qubit_gate(sv, inst.gate, inst.qubits, n)
            else:
                sv = _apply_multi_qubit_gate(sv, inst.gate, inst.qubits, n)

        elapsed = time.perf_counter() - t0
        return SimulatorResult(sv, n, elapsed, seed)


class SimulatorResult:
    """Result of a circuit simulation."""

    def __init__(
        self,
        statevector: np.ndarray,
        n_qubits: int,
        time_seconds: float,
        seed: Optional[int],
    ):
        self.statevector = statevector
        self.n_qubits = n_qubits
        self.time_seconds = time_seconds
        self._rng = np.random.default_rng(seed)

    @property
    def probabilities(self) -> np.ndarray:
        return np.abs(self.statevector) ** 2

    def measure(self, n_shots: int = 1024) -> Dict[str, int]:
        """Shot-based measurement (Born rule, §A.3)."""
        probs = self.probabilities
        outcomes = self._rng.choice(len(probs), size=n_shots, p=probs)
        counts: Dict[str, int] = {}
        for o in outcomes:
            bs = format(o, f"0{self.n_qubits}b")
            counts[bs] = counts.get(bs, 0) + 1
        return counts

    def expectation(self, diagonal_or_matrix: np.ndarray) -> float:
        """Compute ⟨ψ|O|ψ⟩.  Accepts diagonal (1-D) or full matrix."""
        if diagonal_or_matrix.ndim == 1:
            return float(np.real(np.sum(self.probabilities * diagonal_or_matrix)))
        return float(np.real(
            np.vdot(self.statevector, diagonal_or_matrix @ self.statevector)
        ))

    def most_probable(self, top_k: int = 1) -> List[Tuple[str, float]]:
        """Return the *top_k* most probable bit-strings with probabilities."""
        probs = self.probabilities
        indices = np.argsort(probs)[::-1][:top_k]
        return [
            (format(idx, f"0{self.n_qubits}b"), float(probs[idx]))
            for idx in indices
        ]


# ---------------------------------------------------------------------------
# Low-level gate application (§F.1 pseudocode, vectorised NumPy)
# ---------------------------------------------------------------------------

def _apply_single_qubit_gate(
    sv: np.ndarray, gate: np.ndarray, q: int, n: int
) -> np.ndarray:
    """Apply a 2×2 gate to qubit *q* of an n-qubit statevector.

    Uses stride-based indexing to avoid full tensor product.
    O(2^n) time and O(1) extra memory.
    """
    dim = 1 << n
    stride = 1 << q
    g00, g01, g10, g11 = gate[0, 0], gate[0, 1], gate[1, 0], gate[1, 1]

    for block_start in range(0, dim, 2 * stride):
        for i in range(block_start, block_start + stride):
            i0 = i
            i1 = i + stride
            a0, a1 = sv[i0], sv[i1]
            sv[i0] = g00 * a0 + g01 * a1
            sv[i1] = g10 * a0 + g11 * a1
    return sv


def _apply_two_qubit_gate(
    sv: np.ndarray, gate: np.ndarray, q1: int, q2: int, n: int
) -> np.ndarray:
    """Apply a 4×4 gate to qubits *q1*, *q2*.

    Follows §F.1 two-qubit pseudocode with proper qubit ordering.
    """
    dim = 1 << n
    qmin, qmax = min(q1, q2), max(q1, q2)
    stride_min = 1 << qmin
    stride_max = 1 << qmax

    # Determine gate index mapping based on qubit order
    if q1 < q2:
        idx_map = [0, 1, 2, 3]  # standard order
    else:
        idx_map = [0, 2, 1, 3]  # swap control/target

    for i3 in range(0, dim, 2 * stride_max):
        for i2 in range(i3, i3 + stride_max, 2 * stride_min):
            for i1 in range(i2, i2 + stride_min):
                indices = [
                    i1,
                    i1 + stride_min,
                    i1 + stride_max,
                    i1 + stride_min + stride_max,
                ]
                # Reorder indices based on qubit convention
                ordered = [indices[idx_map[k]] for k in range(4)]
                amps = np.array([sv[ordered[k]] for k in range(4)])
                new_amps = gate @ amps
                for k in range(4):
                    sv[ordered[k]] = new_amps[k]
    return sv


def _apply_multi_qubit_gate(
    sv: np.ndarray, gate: np.ndarray, qubits: Tuple[int, ...], n: int
) -> np.ndarray:
    """Apply an arbitrary-size gate via full tensor product expansion.

    Falls back for 3+ qubit gates.  Less efficient but correct.
    """
    nq = len(qubits)
    gate_dim = 1 << nq
    dim = 1 << n

    # Build the full 2^n × 2^n matrix via tensor product and permutation
    full_gate = np.eye(dim, dtype=np.complex128)

    # enumerate all basis states affected
    for basis_in in range(dim):
        # Extract sub-state for target qubits
        sub_in = 0
        for k, q in enumerate(qubits):
            sub_in |= ((basis_in >> q) & 1) << k
        for sub_out in range(gate_dim):
            if gate[sub_out, sub_in] == 0:
                continue
            # Construct the full basis index with sub-state replaced
            basis_out = basis_in
            for k, q in enumerate(qubits):
                bit = (sub_out >> k) & 1
                if bit:
                    basis_out |= (1 << q)
                else:
                    basis_out &= ~(1 << q)
            full_gate[basis_out, basis_in] = gate[sub_out, sub_in]

    # Zero out identity part for affected qubits
    sv_new = np.zeros_like(sv)
    for basis_out in range(dim):
        s = np.complex128(0)
        for basis_in in range(dim):
            # Check if they differ only on target qubits
            mask = 0
            for q in qubits:
                mask |= (1 << q)
            if (basis_out & ~mask) != (basis_in & ~mask):
                continue
            sub_in = 0
            for k, q in enumerate(qubits):
                sub_in |= ((basis_in >> q) & 1) << k
            sub_out = 0
            for k, q in enumerate(qubits):
                sub_out |= ((basis_out >> q) & 1) << k
            s += gate[sub_out, sub_in] * sv[basis_in]
        sv_new[basis_out] = s
    return sv_new
