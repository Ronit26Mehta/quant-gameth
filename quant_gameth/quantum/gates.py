"""
Quantum gate library — exact matrix definitions from Mathematical Foundations §A.2.

All gates are returned as dense NumPy arrays (complex128).  For performance-
critical simulation, the circuit simulator applies them via stride-based
index arithmetic instead of full tensor-product expansion.

Parametric gates:
    RX(θ) = e^{-iθX/2} = cos(θ/2)I - i sin(θ/2)X
    RY(θ) = e^{-iθY/2} = cos(θ/2)I - i sin(θ/2)Y
    RZ(θ) = e^{-iθZ/2} = cos(θ/2)I - i sin(θ/2)Z
"""

from __future__ import annotations

import numpy as np
from typing import Optional

_I = np.eye(2, dtype=np.complex128)

# ---------------------------------------------------------------------------
# Pauli gates (§A.2)
# ---------------------------------------------------------------------------

_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

# ---------------------------------------------------------------------------
# Single-qubit fixed gates
# ---------------------------------------------------------------------------

_H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)

_S = np.array([[1, 0], [0, 1j]], dtype=np.complex128)
_S_DAG = np.array([[1, 0], [0, -1j]], dtype=np.complex128)

_T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex128)
_T_DAG = np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=np.complex128)

_SX = np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=np.complex128) / 2

# ---------------------------------------------------------------------------
# Two-qubit fixed gates (§A.2)
# ---------------------------------------------------------------------------

_CNOT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
], dtype=np.complex128)

_CZ = np.diag([1, 1, 1, -1]).astype(np.complex128)

_SWAP = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
], dtype=np.complex128)

# iSWAP
_ISWAP = np.array([
    [1, 0, 0, 0],
    [0, 0, 1j, 0],
    [0, 1j, 0, 0],
    [0, 0, 0, 1],
], dtype=np.complex128)

# ---------------------------------------------------------------------------
# Three-qubit fixed gates
# ---------------------------------------------------------------------------

_TOFFOLI = np.eye(8, dtype=np.complex128)
_TOFFOLI[6, 6] = 0
_TOFFOLI[7, 7] = 0
_TOFFOLI[6, 7] = 1
_TOFFOLI[7, 6] = 1

_FREDKIN = np.eye(8, dtype=np.complex128)
_FREDKIN[5, 5] = 0
_FREDKIN[6, 6] = 0
_FREDKIN[5, 6] = 1
_FREDKIN[6, 5] = 1


class Gates:
    """Static gate factory.

    Usage::

        g = Gates.H()          # Hadamard
        g = Gates.Rx(np.pi/4)  # Rotation about X
        g = Gates.CNOT()       # Controlled-NOT
    """

    # --- Single-qubit fixed ---

    @staticmethod
    def I() -> np.ndarray:
        """Identity gate."""
        return _I.copy()

    @staticmethod
    def X() -> np.ndarray:
        """Pauli-X (bit flip)."""
        return _X.copy()

    @staticmethod
    def Y() -> np.ndarray:
        """Pauli-Y."""
        return _Y.copy()

    @staticmethod
    def Z() -> np.ndarray:
        """Pauli-Z (phase flip)."""
        return _Z.copy()

    @staticmethod
    def H() -> np.ndarray:
        """Hadamard gate."""
        return _H.copy()

    @staticmethod
    def S() -> np.ndarray:
        """S (phase) gate — sqrt(Z)."""
        return _S.copy()

    @staticmethod
    def Sdg() -> np.ndarray:
        """S-dagger gate."""
        return _S_DAG.copy()

    @staticmethod
    def T() -> np.ndarray:
        """T gate — sqrt(S)."""
        return _T.copy()

    @staticmethod
    def Tdg() -> np.ndarray:
        """T-dagger gate."""
        return _T_DAG.copy()

    @staticmethod
    def SX() -> np.ndarray:
        """√X gate."""
        return _SX.copy()

    # --- Single-qubit parametric (§A.2 rotation gates) ---

    @staticmethod
    def Rx(theta: float) -> np.ndarray:
        """RX(θ) = exp(-iθX/2) = cos(θ/2)I - i·sin(θ/2)X."""
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)

    @staticmethod
    def Ry(theta: float) -> np.ndarray:
        """RY(θ) = exp(-iθY/2) = cos(θ/2)I - i·sin(θ/2)Y."""
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([[c, -s], [s, c]], dtype=np.complex128)

    @staticmethod
    def Rz(theta: float) -> np.ndarray:
        """RZ(θ) = exp(-iθZ/2)."""
        return np.array([
            [np.exp(-1j * theta / 2), 0],
            [0, np.exp(1j * theta / 2)],
        ], dtype=np.complex128)

    @staticmethod
    def Phase(phi: float) -> np.ndarray:
        """Phase gate P(φ) = diag(1, e^{iφ})."""
        return np.array([[1, 0], [0, np.exp(1j * phi)]], dtype=np.complex128)

    @staticmethod
    def U3(theta: float, phi: float, lam: float) -> np.ndarray:
        """General single-qubit unitary U3(θ, φ, λ)."""
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([
            [c, -np.exp(1j * lam) * s],
            [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c],
        ], dtype=np.complex128)

    # --- Two-qubit fixed ---

    @staticmethod
    def CNOT() -> np.ndarray:
        """Controlled-NOT (CX)."""
        return _CNOT.copy()

    @staticmethod
    def CX() -> np.ndarray:
        """Alias for CNOT."""
        return _CNOT.copy()

    @staticmethod
    def CZ() -> np.ndarray:
        """Controlled-Z."""
        return _CZ.copy()

    @staticmethod
    def SWAP() -> np.ndarray:
        """SWAP gate."""
        return _SWAP.copy()

    @staticmethod
    def ISWAP() -> np.ndarray:
        """iSWAP gate."""
        return _ISWAP.copy()

    # --- Two-qubit parametric ---

    @staticmethod
    def CRx(theta: float) -> np.ndarray:
        """Controlled-RX."""
        out = np.eye(4, dtype=np.complex128)
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        out[2, 2] = c
        out[2, 3] = -1j * s
        out[3, 2] = -1j * s
        out[3, 3] = c
        return out

    @staticmethod
    def CRy(theta: float) -> np.ndarray:
        """Controlled-RY."""
        out = np.eye(4, dtype=np.complex128)
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        out[2, 2] = c
        out[2, 3] = -s
        out[3, 2] = s
        out[3, 3] = c
        return out

    @staticmethod
    def CRz(theta: float) -> np.ndarray:
        """Controlled-RZ."""
        out = np.eye(4, dtype=np.complex128)
        out[2, 2] = np.exp(-1j * theta / 2)
        out[3, 3] = np.exp(1j * theta / 2)
        return out

    @staticmethod
    def RZZ(theta: float) -> np.ndarray:
        """RZZ(θ) = exp(-iθ Z⊗Z / 2)."""
        return np.diag([
            np.exp(-1j * theta / 2),
            np.exp(1j * theta / 2),
            np.exp(1j * theta / 2),
            np.exp(-1j * theta / 2),
        ]).astype(np.complex128)

    @staticmethod
    def RXX(theta: float) -> np.ndarray:
        """RXX(θ) = exp(-iθ X⊗X / 2)."""
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([
            [c, 0, 0, -1j * s],
            [0, c, -1j * s, 0],
            [0, -1j * s, c, 0],
            [-1j * s, 0, 0, c],
        ], dtype=np.complex128)

    @staticmethod
    def RYY(theta: float) -> np.ndarray:
        """RYY(θ) = exp(-iθ Y⊗Y / 2)."""
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([
            [c, 0, 0, 1j * s],
            [0, c, -1j * s, 0],
            [0, -1j * s, c, 0],
            [1j * s, 0, 0, c],
        ], dtype=np.complex128)

    # --- Three-qubit ---

    @staticmethod
    def Toffoli() -> np.ndarray:
        """Toffoli (CCX) gate."""
        return _TOFFOLI.copy()

    @staticmethod
    def CCX() -> np.ndarray:
        """Alias for Toffoli."""
        return _TOFFOLI.copy()

    @staticmethod
    def Fredkin() -> np.ndarray:
        """Fredkin (controlled-SWAP) gate."""
        return _FREDKIN.copy()

    @staticmethod
    def CSWAP() -> np.ndarray:
        """Alias for Fredkin."""
        return _FREDKIN.copy()
