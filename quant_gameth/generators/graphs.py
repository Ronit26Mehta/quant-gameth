"""
Graph generators — random, Erdős–Rényi, Barabási–Albert, regular, complete.
"""

from __future__ import annotations

import numpy as np


def generate_graph(
    n_nodes: int,
    graph_type: str = "erdos_renyi",
    p: float = 0.3,
    k: int = 3,
    seed: int = 42,
    weighted: bool = False,
) -> np.ndarray:
    """Generate a random adjacency matrix.

    Parameters
    ----------
    n_nodes : int
    graph_type : str
        ``'erdos_renyi'``, ``'barabasi_albert'``, ``'regular'``,
        ``'complete'``, ``'cycle'``, ``'grid'``.
    p : float
        Edge probability (Erdős–Rényi).
    k : int
        Edges per new node (Barabási–Albert) or degree (regular).
    seed : int
    weighted : bool
        If True, edges have random weights in [0.1, 1.0].
    """
    rng = np.random.default_rng(seed)

    if graph_type == "erdos_renyi":
        adj = _erdos_renyi(n_nodes, p, rng)
    elif graph_type == "barabasi_albert":
        adj = _barabasi_albert(n_nodes, k, rng)
    elif graph_type == "regular":
        adj = _regular(n_nodes, k, rng)
    elif graph_type == "complete":
        adj = np.ones((n_nodes, n_nodes)) - np.eye(n_nodes)
    elif graph_type == "cycle":
        adj = np.zeros((n_nodes, n_nodes))
        for i in range(n_nodes):
            adj[i, (i + 1) % n_nodes] = 1
            adj[(i + 1) % n_nodes, i] = 1
    elif graph_type == "grid":
        side = int(np.ceil(np.sqrt(n_nodes)))
        adj = np.zeros((n_nodes, n_nodes))
        for i in range(n_nodes):
            r, c = divmod(i, side)
            if c + 1 < side and i + 1 < n_nodes:
                adj[i, i + 1] = 1
                adj[i + 1, i] = 1
            if r + 1 < side and i + side < n_nodes:
                adj[i, i + side] = 1
                adj[i + side, i] = 1
    else:
        adj = _erdos_renyi(n_nodes, p, rng)

    if weighted:
        weights = rng.uniform(0.1, 1.0, (n_nodes, n_nodes))
        weights = (weights + weights.T) / 2
        adj = adj * weights

    return adj


def _erdos_renyi(n: int, p: float, rng: np.random.Generator) -> np.ndarray:
    adj = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                adj[i, j] = 1
                adj[j, i] = 1
    return adj


def _barabasi_albert(n: int, m: int, rng: np.random.Generator) -> np.ndarray:
    adj = np.zeros((n, n))
    # Start with m+1 fully connected nodes
    for i in range(min(m + 1, n)):
        for j in range(i + 1, min(m + 1, n)):
            adj[i, j] = 1
            adj[j, i] = 1

    degrees = adj.sum(axis=1)
    for new_node in range(m + 1, n):
        # Preferential attachment
        total_degree = max(degrees[:new_node].sum(), 1)
        probs = degrees[:new_node] / total_degree
        targets = rng.choice(new_node, size=min(m, new_node), replace=False, p=probs)
        for t in targets:
            adj[new_node, t] = 1
            adj[t, new_node] = 1
            degrees[t] += 1
        degrees[new_node] = len(targets)

    return adj


def _regular(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Approximate k-regular graph via random swaps."""
    adj = np.zeros((n, n))
    # Start with cycle + extra edges
    for i in range(n):
        for d in range(1, k // 2 + 1):
            j = (i + d) % n
            adj[i, j] = 1
            adj[j, i] = 1
    return adj
