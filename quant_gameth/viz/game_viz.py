"""
Game theory visualization — payoff matrices, game trees, evolutionary trajectories.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def plot_payoff_matrix(
    payoff_1: np.ndarray,
    payoff_2: np.ndarray,
    title: str = "Payoff Matrix",
    player_names: tuple = ("Player 1", "Player 2"),
    filepath: Optional[str] = None,
) -> Optional[object]:
    """Visualise a bimatrix game as an annotated heatmap."""
    if not HAS_MPL:
        return None

    m, k = payoff_1.shape
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, max(4, m * 0.8)))

    im1 = ax1.imshow(payoff_1, cmap="YlGn", aspect="auto")
    ax1.set_title(f"{player_names[0]}", fontsize=13, fontweight="bold")
    for i in range(m):
        for j in range(k):
            ax1.text(j, i, f"{payoff_1[i, j]:.1f}", ha="center", va="center",
                     fontsize=11, fontweight="bold")
    ax1.set_xlabel("P2 Strategy")
    ax1.set_ylabel("P1 Strategy")

    im2 = ax2.imshow(payoff_2, cmap="YlOrRd", aspect="auto")
    ax2.set_title(f"{player_names[1]}", fontsize=13, fontweight="bold")
    for i in range(m):
        for j in range(k):
            ax2.text(j, i, f"{payoff_2[i, j]:.1f}", ha="center", va="center",
                     fontsize=11, fontweight="bold")
    ax2.set_xlabel("P2 Strategy")

    fig.suptitle(title, fontsize=15, fontweight="bold")
    plt.tight_layout()
    if filepath:
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_evolutionary_trajectory(
    trajectory: List[List[float]],
    strategy_names: Optional[List[str]] = None,
    title: str = "Evolutionary Trajectory",
    filepath: Optional[str] = None,
) -> Optional[object]:
    """Plot population dynamics over time."""
    if not HAS_MPL:
        return None

    data = np.array(trajectory)
    n_strats = data.shape[1]
    if strategy_names is None:
        strategy_names = [f"Strategy {i}" for i in range(n_strats)]

    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(n_strats):
        ax.plot(data[:, i], label=strategy_names[i], linewidth=2)

    ax.set_xlabel("Time Step", fontsize=12)
    ax.set_ylabel("Population Frequency", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if filepath:
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_tournament_results(
    ranking: List[tuple],
    title: str = "Tournament Results",
    filepath: Optional[str] = None,
) -> Optional[object]:
    """Bar chart of tournament rankings."""
    if not HAS_MPL:
        return None

    names = [r[0] for r in ranking]
    scores = [r[1] for r in ranking]

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.5)))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(names)))
    bars = ax.barh(range(len(names)), scores, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlabel("Total Score", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.invert_yaxis()

    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f"{score:.2f}", va="center", fontsize=10)

    plt.tight_layout()
    if filepath:
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig
