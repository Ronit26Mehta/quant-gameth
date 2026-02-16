"""
Serialization utilities — save/load states, circuits, results.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy arrays."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return {"__ndarray__": True, "data": obj.tolist(), "dtype": str(obj.dtype)}
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.complexfloating):
            return {"__complex__": True, "real": float(obj.real), "imag": float(obj.imag)}
        return super().default(obj)


def numpy_decoder(obj: Dict) -> Any:
    """JSON decoder hook for NumPy arrays."""
    if "__ndarray__" in obj:
        return np.array(obj["data"], dtype=obj["dtype"])
    if "__complex__" in obj:
        return complex(obj["real"], obj["imag"])
    return obj


def save_result(result: Any, filepath: str) -> None:
    """Save a result object to JSON."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    data = {}
    for key in ("solution", "energy", "method", "iterations", "time_seconds",
                "converged", "metadata", "strategies", "payoffs",
                "equilibrium_type", "is_pure"):
        if hasattr(result, key):
            val = getattr(result, key)
            if val is not None:
                data[key] = val

    with open(filepath, "w") as f:
        json.dump(data, f, cls=NumpyEncoder, indent=2)


def load_result(filepath: str) -> Dict:
    """Load a result from JSON."""
    with open(filepath, "r") as f:
        return json.load(f, object_hook=numpy_decoder)


def save_circuit(circuit: Any, filepath: str) -> None:
    """Save a quantum circuit specification to JSON."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    data = {
        "n_qubits": circuit.n_qubits,
        "gates": [(g[0], list(g[1]),
                    [p.tolist() if isinstance(p, np.ndarray) else p for p in g[2]])
                   for g in circuit.gates],
    }
    with open(filepath, "w") as f:
        json.dump(data, f, cls=NumpyEncoder, indent=2)


def load_circuit(filepath: str) -> Dict:
    """Load a circuit specification from JSON."""
    with open(filepath, "r") as f:
        return json.load(f, object_hook=numpy_decoder)
