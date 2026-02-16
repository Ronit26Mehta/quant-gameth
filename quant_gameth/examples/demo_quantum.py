"""
demo_quantum.py — Quantum Engine Showcase
==========================================
Run:  python -m quant_gameth.examples.demo_quantum

Demonstrates:
  1. Quantum state creation and manipulation
  2. Gate application and circuit construction
  3. Bell state + entanglement verification
  4. Grover's search for a marked item
  5. QAOA on a small MaxCut instance
  6. VQE ground-state estimation
"""

from __future__ import annotations

import numpy as np


def main() -> None:
    print("=" * 70)
    print("  QUANTUM ENGINE DEMO — quant-gameth")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Quantum State basics
    # ------------------------------------------------------------------
    print("\n▸ 1) Quantum State basics")
    from quant_gameth.quantum.state import QuantumState

    psi = QuantumState.zero(2)
    print(f"  |00⟩ state vector: {psi.vector}")
    print(f"  Probabilities    : {psi.probabilities()}")

    # ------------------------------------------------------------------
    # 2. Circuit: create a Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
    # ------------------------------------------------------------------
    print("\n▸ 2) Bell State construction")
    from quant_gameth.quantum.circuit import QuantumCircuit, Simulator

    qc = QuantumCircuit(2)
    qc.h(0).cx(0, 1)

    sim = Simulator()
    sv = sim.run(qc)
    probs = np.abs(sv) ** 2
    print(f"  Bell state |Φ+⟩ amplitudes: {np.round(sv, 4)}")
    print(f"  Probabilities: |00⟩={probs[0]:.3f}, |01⟩={probs[1]:.3f}, "
          f"|10⟩={probs[2]:.3f}, |11⟩={probs[3]:.3f}")

    # Verify entanglement via partial trace entropy
    bell_state = QuantumState(sv)
    entropy = bell_state.entropy()
    print(f"  Entanglement entropy: {entropy:.4f}  (1.0 = maximally entangled)")

    # ------------------------------------------------------------------
    # 3. Measurement sampling
    # ------------------------------------------------------------------
    print("\n▸ 3) Measurement sampling (1024 shots)")
    from quant_gameth.quantum.measurement import sample_counts

    counts = sample_counts(sv, n_shots=1024, seed=42)
    print(f"  Counts: {counts}")

    # ------------------------------------------------------------------
    # 4. Grover's search
    # ------------------------------------------------------------------
    print("\n▸ 4) Grover's search (4 qubits, target=|1010⟩)")
    from quant_gameth.quantum.grover import grover_search

    target = 0b1010  # target state index = 10
    result = grover_search(n_qubits=4, target_states=[target])
    found = int(np.argmax(np.abs(result.solution) ** 2))
    print(f"  Target: {target} (|{format(target, '04b')}⟩)")
    print(f"  Found : {found} (|{format(found, '04b')}⟩)")
    print(f"  Success probability: {np.abs(result.solution[target])**2:.4f}")
    print(f"  Iterations: {result.iterations}")

    # ------------------------------------------------------------------
    # 5. QAOA on a triangle graph (MaxCut)
    # ------------------------------------------------------------------
    print("\n▸ 5) QAOA MaxCut on triangle graph")
    from quant_gameth.quantum.qaoa import QAOASolver
    from quant_gameth.encoders.qubo import QUBOBuilder

    # Triangle: all edges weight 1
    adj = np.array([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
    ], dtype=float)

    qubo = QUBOBuilder.from_maxcut(adj)
    cost_diag = qubo.to_cost_diagonal()

    qaoa = QAOASolver(n_qubits=3, depth=2)
    qaoa_result = qaoa.solve(cost_diag, seed=42)

    print(f"  Optimal cost (QAOA): {qaoa_result.energy:.4f}")
    print(f"  Converged: {qaoa_result.converged}")
    print(f"  Known optimal cut for triangle: 2")

    # ------------------------------------------------------------------
    # 6. VQE
    # ------------------------------------------------------------------
    print("\n▸ 6) VQE ground-state estimation")
    from quant_gameth.quantum.vqe import VQESolver

    # Simple 2-qubit Hamiltonian: H = Z⊗Z (eigenvalues ±1)
    H = np.diag([1, -1, -1, 1]).astype(float)
    vqe = VQESolver(n_qubits=2, n_layers=2)
    vqe_result = vqe.solve(H, seed=42)

    print(f"  Ground state energy: {vqe_result.energy:.4f}")
    print(f"  Exact ground state : -1.0000")
    print(f"  Iterations: {vqe_result.iterations}")

    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  ✓ Quantum Engine demo complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
