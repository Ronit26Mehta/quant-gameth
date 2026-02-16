"""
demo_maxcut.py — MaxCut Solver Showcase
========================================
Run:  python -m quant_gameth.examples.demo_maxcut

Demonstrates:
  1. Graph generation (Erdős–Rényi)
  2. Brute-force optimal solution (small graphs)
  3. QAOA-based quantum solver
  4. Simulated annealing
  5. Approximation ratio comparison
"""

from __future__ import annotations

import numpy as np


def main() -> None:
    print("=" * 70)
    print("  MAXCUT SOLVER DEMO — quant-gameth")
    print("=" * 70)

    from quant_gameth.generators.graphs import generate_graph
    from quant_gameth.solvers.maxcut import solve_maxcut

    # ------------------------------------------------------------------
    # 1. Small graph — exact comparison
    # ------------------------------------------------------------------
    print("\n▸ 1) Small graph (6 nodes, p=0.5)")
    adj = generate_graph(6, graph_type="erdos_renyi", p=0.5, seed=42)
    n_edges = int(adj.sum() / 2)
    print(f"  Nodes: 6, Edges: {n_edges}")

    # Brute force (optimal)
    bf = solve_maxcut(adj, method="brute_force")
    print(f"\n  Brute Force (exact):")
    print(f"    Cut value : {bf.metadata['cut_value']:.0f}")
    print(f"    Partition : {bf.solution}")
    print(f"    Time      : {bf.time_seconds*1000:.1f}ms")

    # QAOA
    qaoa = solve_maxcut(adj, method="qaoa", qaoa_depth=3, seed=42)
    print(f"\n  QAOA (depth=3):")
    print(f"    Cut value : {qaoa.metadata.get('cut_value', 'N/A')}")
    print(f"    Partition : {qaoa.solution}")
    print(f"    Time      : {qaoa.time_seconds*1000:.1f}ms")

    # Simulated annealing
    sa = solve_maxcut(adj, method="annealing", sa_steps=5000, seed=42)
    print(f"\n  Simulated Annealing:")
    print(f"    Cut value : {sa.metadata['cut_value']:.0f}")
    print(f"    Partition : {sa.solution}")
    print(f"    Time      : {sa.time_seconds*1000:.1f}ms")

    # Approximation ratios
    optimal = bf.metadata["cut_value"]
    if optimal > 0:
        print(f"\n  Approximation ratios (vs brute-force={optimal:.0f}):")
        qaoa_cut = qaoa.metadata.get("cut_value", 0)
        sa_cut = sa.metadata.get("cut_value", 0)
        print(f"    QAOA     : {qaoa_cut/optimal:.4f}")
        print(f"    Annealing: {sa_cut/optimal:.4f}")

    # ------------------------------------------------------------------
    # 2. Larger graph — heuristics only
    # ------------------------------------------------------------------
    print("\n▸ 2) Larger graph (20 nodes, Barabási-Albert)")
    adj_large = generate_graph(20, graph_type="barabasi_albert", k=3, seed=42)
    n_edges_large = int(adj_large.sum() / 2)
    print(f"  Nodes: 20, Edges: {n_edges_large}")

    sa_large = solve_maxcut(adj_large, method="annealing", sa_steps=10000, seed=42)
    print(f"\n  Simulated Annealing:")
    print(f"    Cut value : {sa_large.metadata['cut_value']:.0f}")
    print(f"    Time      : {sa_large.time_seconds*1000:.1f}ms")

    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  ✓ MaxCut demo complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
