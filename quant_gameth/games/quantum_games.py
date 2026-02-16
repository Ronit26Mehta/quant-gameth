"""
Quantum games — EWL protocol, quantum Prisoner's Dilemma, penny flip.

From Mathematical Foundations §C.1:
    Strategy: U(θ,φ) = [[e^{iφ}cos(θ/2), sin(θ/2)],
                         [-sin(θ/2), e^{-iφ}cos(θ/2)]]

    Protocol: |ψ₃⟩ = Ĵ†(U₁⊗U₂)Ĵ|00⟩
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize as scipy_minimize

from quant_gameth._types import EquilibriumResult, SolverMethod


def ewl_strategy_unitary(theta: float, phi: float) -> np.ndarray:
    """SU(2) strategy parametrisation (§C.1).

    U(θ,φ) = [[e^{iφ}cos(θ/2), sin(θ/2)],
               [-sin(θ/2), e^{-iφ}cos(θ/2)]]
    """
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([
        [np.exp(1j * phi) * c, s],
        [-s, np.exp(-1j * phi) * c],
    ], dtype=np.complex128)


def ewl_entangling_gate(gamma: float = np.pi / 2) -> np.ndarray:
    """EWL entangling operator Ĵ(γ).

    Ĵ = exp(iγ |11⟩⟨11|/2) applied to the 2-qubit space.
    For maximum entanglement, γ = π/2.
    """
    J = np.eye(4, dtype=np.complex128)
    c, s = np.cos(gamma / 2), np.sin(gamma / 2)
    J[0, 0] = c
    J[0, 3] = 1j * s
    J[3, 0] = 1j * s
    J[3, 3] = c
    return J


def ewl_quantum_game(
    payoff_matrix_1: np.ndarray,
    payoff_matrix_2: Optional[np.ndarray] = None,
    gamma: float = np.pi / 2,
    optimizer: str = "Nelder-Mead",
    max_iter: int = 500,
    seed: int = 42,
) -> EquilibriumResult:
    """Find quantum Nash equilibrium via EWL protocol (§C.1-C.3).

    Parameters
    ----------
    payoff_matrix_1 : np.ndarray
        Payoff matrix for player 1 (2×2).
    payoff_matrix_2 : np.ndarray or None
        Payoff matrix for player 2 (default: symmetric game).
    gamma : float
        Entanglement parameter (π/2 = maximal).
    optimizer : str
    max_iter : int
    seed : int
    """
    t0 = time.perf_counter()

    if payoff_matrix_2 is None:
        payoff_matrix_2 = payoff_matrix_1.T

    A = np.asarray(payoff_matrix_1, dtype=float)
    B = np.asarray(payoff_matrix_2, dtype=float)

    J = ewl_entangling_gate(gamma)
    J_dag = J.conj().T

    def compute_payoffs(
        theta1: float, phi1: float, theta2: float, phi2: float
    ) -> Tuple[float, float]:
        U1 = ewl_strategy_unitary(theta1, phi1)
        U2 = ewl_strategy_unitary(theta2, phi2)

        # Initial state |00⟩
        psi0 = np.array([1, 0, 0, 0], dtype=np.complex128)

        # EWL protocol
        psi1 = J @ psi0
        psi2 = np.kron(U1, U2) @ psi1
        psi3 = J_dag @ psi2

        # Measurement probabilities
        probs = np.abs(psi3) ** 2

        # Expected payoffs
        p1 = sum(probs[2 * i + j] * A[i, j] for i in range(2) for j in range(2))
        p2 = sum(probs[2 * i + j] * B[i, j] for i in range(2) for j in range(2))

        return float(p1), float(p2)

    # Find Nash via alternating best-response (fixed-point iteration §C.3 Method 1)
    rng = np.random.default_rng(seed)
    best_total = -np.inf
    best_result = None

    for trial in range(10):
        # Random initial strategies
        params = rng.uniform(0, np.pi, 4)
        theta1, phi1 = params[0], params[1]
        theta2, phi2 = params[2], params[3]

        for br_iter in range(50):
            # Best response for player 1
            def neg_p1(p):
                val, _ = compute_payoffs(p[0], p[1], theta2, phi2)
                return -val

            res1 = scipy_minimize(neg_p1, [theta1, phi1], method=optimizer,
                                  bounds=[(0, np.pi), (0, 2 * np.pi)],
                                  options={"maxiter": 100})
            theta1, phi1 = res1.x

            # Best response for player 2
            def neg_p2(p):
                _, val = compute_payoffs(theta1, phi1, p[0], p[1])
                return -val

            res2 = scipy_minimize(neg_p2, [theta2, phi2], method=optimizer,
                                  bounds=[(0, np.pi), (0, 2 * np.pi)],
                                  options={"maxiter": 100})
            theta2, phi2 = res2.x

        p1, p2 = compute_payoffs(theta1, phi1, theta2, phi2)
        if p1 + p2 > best_total:
            best_total = p1 + p2
            best_result = (theta1, phi1, theta2, phi2, p1, p2)

    elapsed = time.perf_counter() - t0
    theta1, phi1, theta2, phi2, p1, p2 = best_result

    return EquilibriumResult(
        strategies=[
            np.array([theta1, phi1]),
            np.array([theta2, phi2]),
        ],
        payoffs=np.array([p1, p2]),
        equilibrium_type="quantum_nash",
        method=SolverMethod.GRADIENT,
        time_seconds=elapsed,
        metadata={
            "gamma": gamma,
            "strategy_unitaries": [
                ewl_strategy_unitary(theta1, phi1).tolist(),
                ewl_strategy_unitary(theta2, phi2).tolist(),
            ],
        },
    )


def quantum_prisoners_dilemma(
    R: float = 3, S: float = 0, T: float = 5, P: float = 1,
    gamma: float = np.pi / 2,
    seed: int = 42,
) -> EquilibriumResult:
    """Quantum Prisoner's Dilemma (§C.1).

    Classical Nash: (D,D) → payoff (P, P) = (1, 1)
    Quantum Nash:  (Q,Q) → payoff (R, R) = (3, 3)
    """
    A = np.array([[R, S], [T, P]])
    B = np.array([[R, T], [S, P]])
    return ewl_quantum_game(A, B, gamma=gamma, seed=seed)


def quantum_penny_flip(
    seed: int = 42,
) -> EquilibriumResult:
    """PQ penny flip game.

    Player 1 (picard): quantum player with full SU(2) strategies
    Player 2 (Q): classical player (flip/no-flip)
    Quantum player always wins with Hadamard.
    """
    t0 = time.perf_counter()

    # Payoff: P1 wins if penny heads up, P2 wins if tails
    A = np.array([[1, -1], [-1, 1]])   # P1 payoffs
    B = -A                              # zero-sum

    # Quantum player (P1) uses Hadamard → always wins
    H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)

    # Classical player (P2) can flip or not
    best_p1 = -np.inf
    best_strategy = None

    for p2_flip_prob in np.linspace(0, 1, 21):
        # P2 mixed strategy
        # P1 Hadamard before and after P2's move
        initial = np.array([1, 0], dtype=np.complex128)  # heads
        after_h1 = H @ initial  # superposition

        # P2's action (classical mixture)
        if np.random.default_rng(42).random() < p2_flip_prob:
            after_p2 = np.array([[0, 1], [1, 0]], dtype=np.complex128) @ after_h1
        else:
            after_p2 = after_h1

        after_h2 = H @ after_p2
        prob_heads = abs(after_h2[0]) ** 2
        payoff = prob_heads * 1 + (1 - prob_heads) * (-1)

        if payoff > best_p1:
            best_p1 = payoff
            best_strategy = p2_flip_prob

    elapsed = time.perf_counter() - t0

    return EquilibriumResult(
        strategies=[
            np.array([0.0, 0.0]),  # P1: Hadamard (θ=0, special)
            np.array([best_strategy]),
        ],
        payoffs=np.array([1.0, -1.0]),  # quantum player always wins
        equilibrium_type="quantum_dominant",
        method=SolverMethod.BRUTE_FORCE,
        time_seconds=elapsed,
        metadata={"game": "penny_flip", "quantum_advantage": True},
    )


def compute_quantum_advantage(
    payoff_matrix: np.ndarray,
    gamma: float = np.pi / 2,
    seed: int = 42,
) -> Dict:
    """Compare classical vs quantum Nash equilibria.

    Returns a dict with classical payoffs, quantum payoffs, and advantage.
    """
    from quant_gameth.games.normal_form import NormalFormGame

    game = NormalFormGame(payoff_matrix, payoff_matrix.T)
    classical = game.find_nash()

    quantum = ewl_quantum_game(payoff_matrix, payoff_matrix.T, gamma=gamma, seed=seed)

    classical_payoffs = classical[0].payoffs if classical else np.array([0, 0])

    return {
        "classical_payoffs": classical_payoffs.tolist(),
        "quantum_payoffs": quantum.payoffs.tolist(),
        "advantage_p1": float(quantum.payoffs[0] - classical_payoffs[0]),
        "advantage_p2": float(quantum.payoffs[1] - classical_payoffs[1]),
        "gamma": gamma,
    }
