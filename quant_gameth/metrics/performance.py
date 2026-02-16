"""
Performance metrics — timing, memory profiling, approximation ratios, optimality gaps.

Provides decorators and context managers for non-intrusive profiling,
plus aggregation utilities for analysing solver quality.
"""

from __future__ import annotations

import functools
import gc
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TimingRecord:
    """Single timing measurement."""
    label: str
    wall_seconds: float
    cpu_seconds: float
    n_calls: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemorySnapshot:
    """Memory usage at a point in time."""
    label: str
    rss_bytes: int          # resident set size (0 if unavailable)
    python_objects: int     # count of tracked Python objects
    numpy_arrays_mb: float  # rough sum of NumPy array memory


@dataclass
class QualityMetrics:
    """Solution quality metrics for a given solver run."""
    approximation_ratio: Optional[float] = None    # solution / optimal
    optimality_gap: Optional[float] = None         # |solution - optimal|
    feasibility: bool = True                       # constraints satisfied?
    constraint_violations: int = 0
    relative_error: Optional[float] = None         # |sol - opt| / |opt|


# ---------------------------------------------------------------------------
# PerformanceTracker
# ---------------------------------------------------------------------------

class PerformanceTracker:
    """Collect timing, memory, and quality metrics across solver runs.

    Usage::

        tracker = PerformanceTracker()

        with tracker.time("qaoa_solve"):
            result = qaoa.solve(...)

        tracker.record_quality("qaoa_solve", result.energy, optimal=-5.2)
        report = tracker.summary()
    """

    def __init__(self) -> None:
        self._timings: Dict[str, List[TimingRecord]] = {}
        self._memory: Dict[str, List[MemorySnapshot]] = {}
        self._quality: Dict[str, List[QualityMetrics]] = {}

    # ── Timing ──────────────────────────────────────────────────────────

    @contextmanager
    def time(self, label: str):
        """Context manager that records wall-clock and CPU time."""
        gc.collect()
        t_wall_start = time.perf_counter()
        t_cpu_start = time.process_time()
        try:
            yield
        finally:
            wall = time.perf_counter() - t_wall_start
            cpu = time.process_time() - t_cpu_start
            rec = TimingRecord(label=label, wall_seconds=wall, cpu_seconds=cpu)
            self._timings.setdefault(label, []).append(rec)

    def timer(self, label: Optional[str] = None) -> Callable:
        """Decorator that times every call to a function.

        Usage::

            @tracker.timer()
            def my_function(...):
                ...
        """
        def decorator(fn: Callable) -> Callable:
            tag = label or fn.__qualname__

            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                with self.time(tag):
                    return fn(*args, **kwargs)
            return wrapper
        return decorator

    # ── Memory ──────────────────────────────────────────────────────────

    def snapshot_memory(self, label: str) -> MemorySnapshot:
        """Take a memory snapshot right now."""
        gc.collect()

        # RSS via platform-specific approaches
        rss = 0
        try:
            import psutil
            rss = psutil.Process().memory_info().rss
        except ImportError:
            pass

        # Count tracked Python objects
        n_objects = len(gc.get_objects())

        # Estimate NumPy memory
        numpy_mb = 0.0
        for obj in gc.get_objects():
            if isinstance(obj, np.ndarray):
                numpy_mb += obj.nbytes / (1024 * 1024)

        snap = MemorySnapshot(
            label=label,
            rss_bytes=rss,
            python_objects=n_objects,
            numpy_arrays_mb=round(numpy_mb, 2),
        )
        self._memory.setdefault(label, []).append(snap)
        return snap

    # ── Quality ─────────────────────────────────────────────────────────

    def record_quality(
        self,
        label: str,
        solution_value: float,
        optimal_value: Optional[float] = None,
        feasible: bool = True,
        constraint_violations: int = 0,
    ) -> QualityMetrics:
        """Record solution quality for a solver run."""
        q = QualityMetrics(feasibility=feasible,
                           constraint_violations=constraint_violations)

        if optimal_value is not None and optimal_value != 0:
            q.approximation_ratio = solution_value / optimal_value
            q.optimality_gap = abs(solution_value - optimal_value)
            q.relative_error = abs(solution_value - optimal_value) / abs(optimal_value)

        self._quality.setdefault(label, []).append(q)
        return q

    # ── Aggregation ─────────────────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        """Aggregate all recorded metrics into a summary dict."""
        report: Dict[str, Any] = {}

        for label, records in self._timings.items():
            walls = [r.wall_seconds for r in records]
            cpus = [r.cpu_seconds for r in records]
            report.setdefault(label, {})["timing"] = {
                "n_runs": len(records),
                "wall_mean_s": float(np.mean(walls)),
                "wall_std_s": float(np.std(walls)),
                "wall_min_s": float(np.min(walls)),
                "wall_max_s": float(np.max(walls)),
                "cpu_mean_s": float(np.mean(cpus)),
            }

        for label, snaps in self._memory.items():
            report.setdefault(label, {})["memory"] = {
                "rss_bytes": [s.rss_bytes for s in snaps],
                "numpy_mb": [s.numpy_arrays_mb for s in snaps],
                "python_objects": [s.python_objects for s in snaps],
            }

        for label, quals in self._quality.items():
            ratios = [q.approximation_ratio for q in quals
                      if q.approximation_ratio is not None]
            gaps = [q.optimality_gap for q in quals
                    if q.optimality_gap is not None]
            report.setdefault(label, {})["quality"] = {
                "n_runs": len(quals),
                "feasible_rate": sum(1 for q in quals if q.feasibility) / max(len(quals), 1),
                "approx_ratio_mean": float(np.mean(ratios)) if ratios else None,
                "optimality_gap_mean": float(np.mean(gaps)) if gaps else None,
            }

        return report

    def reset(self) -> None:
        """Clear all recorded metrics."""
        self._timings.clear()
        self._memory.clear()
        self._quality.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Export all raw records as a serialisable dict."""
        return {
            "timings": {
                k: [{"wall": r.wall_seconds, "cpu": r.cpu_seconds} for r in v]
                for k, v in self._timings.items()
            },
            "quality": {
                k: [{"approx": q.approximation_ratio, "gap": q.optimality_gap,
                      "feasible": q.feasibility} for q in v]
                for k, v in self._quality.items()
            },
        }


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_global_tracker = PerformanceTracker()


def get_global_tracker() -> PerformanceTracker:
    """Return the global singleton performance tracker."""
    return _global_tracker
