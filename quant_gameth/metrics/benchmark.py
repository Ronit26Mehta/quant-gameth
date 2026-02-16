"""
Benchmark suite — run solvers across problem families, export JSON/CSV.

Automates comparative benchmarking for the framework:
    1. Generate problem instances (varying size / density / difficulty)
    2. Run each configured solver method
    3. Collect timing, quality, and scaling data
    4. Export results for downstream analysis
"""

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from quant_gameth._types import BenchmarkResult, SolverResult
from quant_gameth.metrics.performance import PerformanceTracker, QualityMetrics


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark experiment."""
    problem_name: str
    sizes: List[int]                            # problem sizes to sweep
    methods: List[str]                          # solver method names
    n_repeats: int = 3                          # repetitions per (size, method)
    seed_base: int = 42
    timeout_seconds: float = 300.0              # per-run timeout
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# BenchmarkSuite
# ---------------------------------------------------------------------------

class BenchmarkSuite:
    """Orchestrate benchmarks across problems, sizes, and methods.

    Usage::

        suite = BenchmarkSuite()

        suite.register("maxcut", generate_fn=..., solve_fn=...)
        suite.register("tsp",    generate_fn=..., solve_fn=...)

        results = suite.run(BenchmarkConfig(
            problem_name="maxcut",
            sizes=[6, 8, 10, 12],
            methods=["qaoa", "annealing", "brute_force"],
        ))

        suite.export_json(results, "benchmark_results.json")
        suite.export_csv(results,  "benchmark_results.csv")
    """

    def __init__(self) -> None:
        self._registry: Dict[str, Dict[str, Callable]] = {}
        self._tracker = PerformanceTracker()

    # ── Registration ────────────────────────────────────────────────────

    def register(
        self,
        problem_name: str,
        generate_fn: Callable[[int, int], Any],
        solve_fn: Callable[[Any, str, int], SolverResult],
        optimal_fn: Optional[Callable[[Any], float]] = None,
    ) -> None:
        """Register a problem family.

        Parameters
        ----------
        problem_name : str
        generate_fn : callable
            ``generate_fn(size: int, seed: int) -> problem_instance``
        solve_fn : callable
            ``solve_fn(instance, method: str, seed: int) -> SolverResult``
        optimal_fn : callable or None
            ``optimal_fn(instance) -> float`` — known optimal value (for
            approximation ratio). ``None`` if unavailable.
        """
        self._registry[problem_name] = {
            "generate": generate_fn,
            "solve": solve_fn,
            "optimal": optimal_fn,
        }

    def register_builtin(self, problem_name: str) -> None:
        """Register one of the built-in problem families.

        Supported: ``'maxcut'``, ``'graph_coloring'``, ``'knapsack'``,
        ``'tsp'``, ``'portfolio'``, ``'sudoku'``, ``'nqueens'``.
        """
        if problem_name == "maxcut":
            from quant_gameth.generators.graphs import generate_graph
            from quant_gameth.solvers.maxcut import solve_maxcut

            self.register(
                "maxcut",
                generate_fn=lambda size, seed: generate_graph(size, seed=seed),
                solve_fn=lambda inst, method, seed: solve_maxcut(
                    inst, method=method, seed=seed),
            )

        elif problem_name == "graph_coloring":
            from quant_gameth.generators.graphs import generate_graph
            from quant_gameth.solvers.graph_coloring import solve_graph_coloring

            self.register(
                "graph_coloring",
                generate_fn=lambda size, seed: generate_graph(size, seed=seed),
                solve_fn=lambda inst, method, seed: solve_graph_coloring(
                    inst, method=method, seed=seed),
            )

        elif problem_name == "knapsack":
            from quant_gameth.solvers.knapsack import solve_knapsack

            def gen_knapsack(size: int, seed: int) -> Dict:
                rng = np.random.default_rng(seed)
                values = rng.uniform(1, 100, size)
                weights = rng.uniform(1, 50, size)
                capacity = float(weights.sum() * 0.5)
                return {"values": values, "weights": weights, "capacity": capacity}

            self.register(
                "knapsack",
                generate_fn=gen_knapsack,
                solve_fn=lambda inst, method, seed: solve_knapsack(
                    inst["values"], inst["weights"], inst["capacity"],
                    method=method, seed=seed),
            )

        elif problem_name == "tsp":
            from quant_gameth.solvers.tsp import solve_tsp

            def gen_tsp(size: int, seed: int) -> np.ndarray:
                rng = np.random.default_rng(seed)
                coords = rng.uniform(0, 100, (size, 2))
                dist = np.zeros((size, size))
                for i in range(size):
                    for j in range(size):
                        dist[i, j] = np.linalg.norm(coords[i] - coords[j])
                return dist

            self.register(
                "tsp",
                generate_fn=gen_tsp,
                solve_fn=lambda inst, method, seed: solve_tsp(
                    inst, method=method, seed=seed),
            )

        elif problem_name == "nqueens":
            from quant_gameth.solvers.nqueens import solve_nqueens

            self.register(
                "nqueens",
                generate_fn=lambda size, seed: size,
                solve_fn=lambda inst, method, seed: solve_nqueens(
                    inst, method=method, seed=seed),
            )

    # ── Execution ───────────────────────────────────────────────────────

    def run(self, config: BenchmarkConfig) -> List[BenchmarkResult]:
        """Execute the benchmark sweep.

        Returns one ``BenchmarkResult`` per problem size.
        """
        if config.problem_name not in self._registry:
            raise KeyError(f"Problem '{config.problem_name}' not registered. "
                           f"Available: {list(self._registry.keys())}")

        entry = self._registry[config.problem_name]
        generate_fn = entry["generate"]
        solve_fn = entry["solve"]
        optimal_fn = entry.get("optimal")

        results: List[BenchmarkResult] = []

        for size in config.sizes:
            bench = BenchmarkResult(
                problem_name=config.problem_name,
                problem_size=size,
            )

            for method in config.methods:
                method_results: List[SolverResult] = []

                for rep in range(config.n_repeats):
                    seed = config.seed_base + size * 1000 + rep
                    instance = generate_fn(size, seed)

                    label = f"{config.problem_name}_{method}_n{size}_r{rep}"
                    try:
                        with self._tracker.time(label):
                            result = solve_fn(instance, method, seed)

                        # Record quality if optimal is known
                        if optimal_fn is not None:
                            opt = optimal_fn(instance)
                            self._tracker.record_quality(
                                label, result.energy, opt,
                                feasible=result.converged,
                                constraint_violations=result.constraint_violations,
                            )

                        method_results.append(result)

                    except Exception as exc:
                        # Record failure but don't crash the suite
                        fail_result = SolverResult(
                            solution=np.array([]),
                            energy=float("inf"),
                            method=result.method if 'result' in dir() else None,
                            converged=False,
                            metadata={"error": str(exc)},
                        )
                        method_results.append(fail_result)

                # Aggregate across repeats
                if method_results:
                    best = min(method_results, key=lambda r: r.energy)
                    avg_time = float(np.mean([r.time_seconds for r in method_results]))
                    best.metadata["avg_time_seconds"] = avg_time
                    best.metadata["n_repeats"] = config.n_repeats
                    bench.solver_results[method] = best

            results.append(bench)

        return results

    # ── Export ───────────────────────────────────────────────────────────

    @staticmethod
    def export_json(results: List[BenchmarkResult], filepath: str) -> None:
        """Export benchmark results to JSON."""
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

        data = []
        for bench in results:
            entry: Dict[str, Any] = {
                "problem": bench.problem_name,
                "size": bench.problem_size,
                "methods": {},
            }
            for method, sr in bench.solver_results.items():
                entry["methods"][method] = {
                    "energy": float(sr.energy),
                    "time_s": sr.time_seconds,
                    "converged": sr.converged,
                    "iterations": sr.iterations,
                    "metadata": {k: v for k, v in sr.metadata.items()
                                 if not isinstance(v, np.ndarray)},
                }
            data.append(entry)

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @staticmethod
    def export_csv(results: List[BenchmarkResult], filepath: str) -> None:
        """Export benchmark results to CSV (one row per size × method)."""
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

        rows: List[Dict[str, Any]] = []
        for bench in results:
            for method, sr in bench.solver_results.items():
                rows.append({
                    "problem": bench.problem_name,
                    "size": bench.problem_size,
                    "method": method,
                    "energy": sr.energy,
                    "time_seconds": sr.time_seconds,
                    "iterations": sr.iterations,
                    "converged": sr.converged,
                    "constraint_violations": sr.constraint_violations,
                })

        if rows:
            with open(filepath, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

    def get_tracker(self) -> PerformanceTracker:
        """Access the internal performance tracker for detailed metrics."""
        return self._tracker
