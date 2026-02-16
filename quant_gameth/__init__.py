"""
quant-gameth: A Quantum-Game Theory Framework for Complex Problem Solving
=========================================================================

Unifies quantum computing simulation, game theory, and combinatorial
optimization into a single, hardware-agnostic Python package.

Subpackages
-----------
quantum   : Quantum state simulation, gates, circuits, QAOA, VQE, Grover, annealing
games     : Normal-form, extensive-form, evolutionary, mechanism design, quantum games
encoders  : QUBO/Ising builder, constraint compiler, game-optimization bridge
solvers   : Sudoku, MaxCut, TSP, N-Queens, knapsack, portfolio, graph coloring
generators: Puzzle, game, graph, and market scenario generators
viz       : Quantum state, game tree, solver convergence visualization
metrics   : Performance profiling, benchmark suite
backends  : Classical (NumPy), GPU (CuPy), hybrid execution backends
"""

__version__ = "0.1.0"
__author__ = "Quantum Game Theory Research"

from quant_gameth._types import (
    SolverResult,
    EquilibriumResult,
    GameDescription,
    EncodedProblem,
    GameType,
    SolverMethod,
    Backend,
    AnsatzType,
    ConstraintType,
    OptimizationSense,
)
