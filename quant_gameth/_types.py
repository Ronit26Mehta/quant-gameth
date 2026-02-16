"""
Core type definitions for the quant-gameth framework.

All primary data classes, enumerations, and type aliases used across
the entire package are defined here for consistency and import clarity.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class GameType(enum.Enum):
    """Classification of game types."""
    NORMAL_FORM = "normal_form"
    EXTENSIVE_FORM = "extensive_form"
    REPEATED = "repeated"
    EVOLUTIONARY = "evolutionary"
    COOPERATIVE = "cooperative"
    QUANTUM = "quantum"
    MECHANISM = "mechanism"


class SolverMethod(enum.Enum):
    """Algorithm used to solve a problem."""
    QAOA = "qaoa"
    VQE = "vqe"
    GROVER = "grover"
    ANNEALING = "annealing"
    CLASSICAL_SA = "classical_sa"
    BACKTRACKING = "backtracking"
    DYNAMIC_PROGRAMMING = "dp"
    MINIMAX = "minimax"
    BRUTE_FORCE = "brute_force"
    LINEAR_PROGRAMMING = "lp"
    SUPPORT_ENUMERATION = "support_enumeration"
    LEMKE_HOWSON = "lemke_howson"
    VERTEX_ENUMERATION = "vertex_enumeration"
    REPLICATOR = "replicator"
    MORAN = "moran"
    GRADIENT = "gradient"
    SIMULATED_ANNEALING = "simulated_annealing"
    GREEDY = "greedy"
    LOCAL_SEARCH = "local_search"
    ANALYTICAL = "analytical"
    BRANCH_AND_BOUND = "branch_and_bound"
    CONSTRAINT_PROPAGATION = "constraint_propagation"


class Backend(enum.Enum):
    """Execution backend."""
    CLASSICAL = "classical"
    GPU = "gpu"
    HYBRID = "hybrid"


class AnsatzType(enum.Enum):
    """Variational ansatz type for quantum circuits."""
    HEA = "hardware_efficient"
    QAOA_STANDARD = "qaoa_standard"
    QAOA_PLUS = "qaoa_plus"           # with custom mixers
    UCC = "unitary_coupled_cluster"


class ConstraintType(enum.Enum):
    """Constraint types for QUBO encoding."""
    EXACTLY_ONE = "exactly_one"
    AT_MOST_ONE = "at_most_one"
    ALL_DIFFERENT = "all_different"
    EQUALITY = "equality"
    INEQUALITY_LEQ = "inequality_leq"
    INEQUALITY_GEQ = "inequality_geq"
    LOGICAL_AND = "logical_and"
    LOGICAL_OR = "logical_or"


class OptimizationSense(enum.Enum):
    """Optimization direction."""
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class SolverResult:
    """Result of a solver run.

    Attributes
    ----------
    solution : np.ndarray
        The solution vector (binary or continuous depending on problem).
    energy : float
        Objective function value at the solution.
    method : SolverMethod
        Algorithm used.
    iterations : int
        Number of iterations / function evaluations.
    time_seconds : float
        Wall-clock solving time in seconds.
    converged : bool
        Whether the algorithm converged to a satisfactory solution.
    constraint_violations : int
        Number of violated constraints (0 for feasible solutions).
    history : List[float]
        Objective value history across iterations (for convergence plots).
    metadata : Dict[str, Any]
        Additional solver-specific metadata (e.g., QAOA parameters).
    """
    solution: np.ndarray
    energy: float
    method: SolverMethod
    iterations: int = 0
    time_seconds: float = 0.0
    converged: bool = False
    constraint_violations: int = 0
    history: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_bitstring(self) -> str:
        """Convert binary solution to a bit-string representation."""
        return "".join(str(int(round(b))) for b in self.solution)

    @property
    def approximation_ratio(self) -> Optional[float]:
        """Return approximation ratio if optimal is known."""
        opt = self.metadata.get("optimal_value")
        if opt is not None and opt != 0:
            return self.energy / opt
        return None


@dataclass
class EquilibriumResult:
    """Result of a game equilibrium computation.

    Attributes
    ----------
    strategies : List[np.ndarray]
        Mixed/quantum strategy for each player (probability vectors or
        density matrices).
    payoffs : np.ndarray
        Expected payoff for each player at equilibrium.
    equilibrium_type : str
        Type of equilibrium found (e.g., 'nash', 'subgame_perfect',
        'evolutionary_stable', 'quantum_nash').
    method : SolverMethod
        Algorithm used to find the equilibrium.
    is_pure : bool
        Whether the equilibrium uses pure strategies only.
    time_seconds : float
        Computation time.
    metadata : Dict[str, Any]
        Extra information.
    """
    strategies: List[np.ndarray]
    payoffs: np.ndarray
    equilibrium_type: str = "nash"
    method: SolverMethod = SolverMethod.SUPPORT_ENUMERATION
    is_pure: bool = False
    time_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GameDescription:
    """High-level description of a game.

    Attributes
    ----------
    game_type : GameType
        Classification of the game.
    n_players : int
        Number of players.
    n_strategies : Union[int, List[int]]
        Number of strategies per player (int if same for all).
    payoff_matrices : List[np.ndarray]
        Payoff matrix for each player.  For an m×k game with 2 players,
        ``payoff_matrices[0]`` has shape ``(m, k)`` and
        ``payoff_matrices[1]`` has shape ``(m, k)``.
    name : str
        Human-readable name of the game.
    metadata : Dict[str, Any]
        Additional game metadata (e.g., discount factors for repeated games).
    """
    game_type: GameType
    n_players: int
    n_strategies: Union[int, List[int]]
    payoff_matrices: List[np.ndarray]
    name: str = "unnamed_game"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EncodedProblem:
    """Problem encoded for quantum/hybrid solution.

    Attributes
    ----------
    qubo_matrix : Optional[np.ndarray]
        The Q matrix for QUBO formulation.
    ising_h : Optional[np.ndarray]
        Linear Ising coefficients (bias field).
    ising_J : Optional[np.ndarray]
        Coupling matrix for Ising model.
    n_variables : int
        Number of binary decision variables.
    n_logical_qubits : int
        Number of qubits required for quantum encoding.
    offset : float
        Constant offset in the objective function.
    variable_labels : Optional[List[str]]
        Human-readable labels for each variable.
    constraints_count : int
        Number of encoded constraints.
    problem_type : str
        Original problem type (e.g., 'sudoku', 'maxcut', 'game').
    metadata : Dict[str, Any]
        Encoding metadata (penalty weights, slack counts, etc.).
    """
    qubo_matrix: Optional[np.ndarray] = None
    ising_h: Optional[np.ndarray] = None
    ising_J: Optional[np.ndarray] = None
    n_variables: int = 0
    n_logical_qubits: int = 0
    offset: float = 0.0
    variable_labels: Optional[List[str]] = None
    constraints_count: int = 0
    problem_type: str = "generic"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Single benchmark run result.

    Attributes
    ----------
    problem_name : str
        Name of the benchmark problem.
    problem_size : int
        Problem size parameter (number of variables / nodes / cells).
    solver_results : Dict[str, SolverResult]
        Results keyed by method name.
    """
    problem_name: str
    problem_size: int
    solver_results: Dict[str, SolverResult] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Type Aliases
# ---------------------------------------------------------------------------

PayoffMatrix = np.ndarray          # shape (n_strategies_p1, n_strategies_p2)
QUBOMatrix = np.ndarray            # shape (n, n) upper-triangular
IsingCoefficients = Tuple[np.ndarray, np.ndarray]  # (h, J)
Statevector = np.ndarray           # complex128 array of size 2^n
DensityMatrix = np.ndarray         # complex128 array of size (2^n, 2^n)
BitString = str                    # e.g. "01101"
