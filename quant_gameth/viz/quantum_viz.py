"""
Quantum state visualization — Bloch sphere, circuit diagrams, state tomography.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def plot_bloch_sphere(
    states: List[np.ndarray],
    labels: Optional[List[str]] = None,
    title: str = "Bloch Sphere",
    filepath: Optional[str] = None,
) -> Optional[object]:
    """Plot quantum states on the Bloch sphere.

    Parameters
    ----------
    states : list of np.ndarray
        List of 2-element statevectors.
    labels : list of str or None
    title : str
    filepath : str or None
        If provided, save figure to file.
    """
    if not HAS_MPL:
        return None

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Draw sphere wireframe
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.05, color="cyan")

    # Draw axes
    ax.plot([-1.3, 1.3], [0, 0], [0, 0], "k-", alpha=0.3)
    ax.plot([0, 0], [-1.3, 1.3], [0, 0], "k-", alpha=0.3)
    ax.plot([0, 0], [0, 0], [-1.3, 1.3], "k-", alpha=0.3)
    ax.text(1.4, 0, 0, "x", fontsize=12)
    ax.text(0, 1.4, 0, "y", fontsize=12)
    ax.text(0, 0, 1.4, "|0⟩", fontsize=12, color="blue")
    ax.text(0, 0, -1.4, "|1⟩", fontsize=12, color="red")

    colors = plt.cm.Set1(np.linspace(0, 1, max(len(states), 1)))

    for idx, state in enumerate(states):
        # Compute Bloch vector
        if len(state) != 2:
            continue
        a, b = state[0], state[1]
        bx = 2 * np.real(np.conj(a) * b)
        by = 2 * np.imag(np.conj(a) * b)
        bz = float(np.abs(a) ** 2 - np.abs(b) ** 2)

        label = labels[idx] if labels else f"ψ{idx}"
        ax.scatter([bx], [by], [bz], s=100, color=colors[idx], label=label, zorder=5)
        ax.plot([0, bx], [0, by], [0, bz], color=colors[idx], linewidth=2)

    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper left")

    if filepath:
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_statevector(
    statevector: np.ndarray,
    title: str = "Statevector",
    filepath: Optional[str] = None,
) -> Optional[object]:
    """Bar chart of statevector amplitudes and phases."""
    if not HAS_MPL:
        return None

    n = len(statevector)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(8, n * 0.4), 8))

    indices = np.arange(n)
    n_qubits = int(np.log2(n)) if n > 0 else 0
    labels_list = [f"|{format(i, f'0{n_qubits}b')}⟩" for i in range(n)]

    # Probability amplitudes
    probs = np.abs(statevector) ** 2
    ax1.bar(indices, probs, color="steelblue", alpha=0.8)
    ax1.set_ylabel("Probability", fontsize=12)
    ax1.set_title(f"{title} — Probabilities", fontsize=13, fontweight="bold")
    ax1.set_xticks(indices)
    ax1.set_xticklabels(labels_list, rotation=45, ha="right", fontsize=8)

    # Phases
    phases = np.angle(statevector)
    phase_mask = probs > 1e-10
    colors = ["steelblue" if m else "lightgray" for m in phase_mask]
    ax2.bar(indices, phases / np.pi, color=colors, alpha=0.8)
    ax2.set_ylabel("Phase (× π)", fontsize=12)
    ax2.set_title(f"{title} — Phases", fontsize=13, fontweight="bold")
    ax2.set_xticks(indices)
    ax2.set_xticklabels(labels_list, rotation=45, ha="right", fontsize=8)

    plt.tight_layout()
    if filepath:
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_density_matrix(
    rho: np.ndarray,
    title: str = "Density Matrix",
    filepath: Optional[str] = None,
) -> Optional[object]:
    """Heatmap of density matrix (real and imaginary parts)."""
    if not HAS_MPL:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    im1 = ax1.imshow(np.real(rho), cmap="RdBu_r", aspect="equal", vmin=-1, vmax=1)
    ax1.set_title(f"{title} (Real)", fontsize=13, fontweight="bold")
    plt.colorbar(im1, ax=ax1, fraction=0.046)

    im2 = ax2.imshow(np.imag(rho), cmap="PiYG", aspect="equal", vmin=-1, vmax=1)
    ax2.set_title(f"{title} (Imaginary)", fontsize=13, fontweight="bold")
    plt.colorbar(im2, ax=ax2, fraction=0.046)

    plt.tight_layout()
    if filepath:
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig
