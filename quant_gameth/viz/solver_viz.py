"""
Solver visualization — convergence plots, solution diagrams.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def plot_convergence(
    energies: List[float],
    title: str = "Solver Convergence",
    xlabel: str = "Iteration",
    ylabel: str = "Energy / Cost",
    filepath: Optional[str] = None,
) -> Optional[object]:
    """Plot energy/cost convergence over iterations."""
    if not HAS_MPL:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(energies, linewidth=2, color="darkblue")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Annotate best value
    best_idx = int(np.argmin(energies))
    ax.annotate(
        f"Best: {energies[best_idx]:.4f}",
        xy=(best_idx, energies[best_idx]),
        xytext=(best_idx + len(energies) * 0.05, energies[best_idx] * 1.1),
        fontsize=11,
        arrowprops=dict(arrowstyle="->", color="red"),
        color="red",
        fontweight="bold",
    )

    plt.tight_layout()
    if filepath:
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_graph_solution(
    adjacency: np.ndarray,
    partition: np.ndarray,
    title: str = "Graph Partition",
    filepath: Optional[str] = None,
) -> Optional[object]:
    """Visualise graph with node coloring (MaxCut/coloring solution)."""
    if not HAS_MPL:
        return None

    n = len(adjacency)
    fig, ax = plt.subplots(figsize=(8, 8))

    # Circular layout
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x_pos = np.cos(angles)
    y_pos = np.sin(angles)

    # Draw edges
    for i in range(n):
        for j in range(i + 1, n):
            if adjacency[i, j] != 0:
                color = "red" if partition[i] != partition[j] else "lightgray"
                lw = 2 if partition[i] != partition[j] else 0.5
                ax.plot([x_pos[i], x_pos[j]], [y_pos[i], y_pos[j]],
                        color=color, linewidth=lw, alpha=0.6)

    # Draw nodes
    unique_parts = np.unique(partition)
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(unique_parts), 2)))
    color_map = {p: colors[idx] for idx, p in enumerate(unique_parts)}

    for i in range(n):
        ax.scatter(x_pos[i], y_pos[i], s=300, color=color_map[partition[i]],
                   edgecolors="black", linewidth=2, zorder=5)
        ax.text(x_pos[i], y_pos[i], str(i), ha="center", va="center",
                fontsize=10, fontweight="bold")

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout()
    if filepath:
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_tsp_tour(
    coordinates: np.ndarray,
    tour: np.ndarray,
    title: str = "TSP Tour",
    filepath: Optional[str] = None,
) -> Optional[object]:
    """Plot TSP tour on 2D plane."""
    if not HAS_MPL:
        return None

    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw tour
    for i in range(len(tour)):
        j = (i + 1) % len(tour)
        ax.plot(
            [coordinates[tour[i], 0], coordinates[tour[j], 0]],
            [coordinates[tour[i], 1], coordinates[tour[j], 1]],
            "b-", linewidth=2, alpha=0.7,
        )

    ax.scatter(coordinates[:, 0], coordinates[:, 1], s=100, c="red",
               edgecolors="black", linewidth=2, zorder=5)
    for i in range(len(coordinates)):
        ax.text(coordinates[i, 0] + 0.02, coordinates[i, 1] + 0.02,
                str(i), fontsize=9, fontweight="bold")

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if filepath:
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_portfolio_frontier(
    returns: np.ndarray,
    risks: np.ndarray,
    weights: Optional[np.ndarray] = None,
    optimal_idx: Optional[int] = None,
    title: str = "Efficient Frontier",
    filepath: Optional[str] = None,
) -> Optional[object]:
    """Plot Markowitz efficient frontier."""
    if not HAS_MPL:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(risks, returns, c=returns / np.maximum(np.sqrt(risks), 1e-10),
               cmap="viridis", s=30, alpha=0.7, label="Portfolios")

    if optimal_idx is not None:
        ax.scatter(risks[optimal_idx], returns[optimal_idx], c="red", s=200,
                   marker="*", zorder=5, label="Optimal")

    ax.set_xlabel("Risk (Variance)", fontsize=12)
    ax.set_ylabel("Expected Return", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if filepath:
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig
