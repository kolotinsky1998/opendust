"""Plotting helpers for force validation artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_force_scatter(reference_forces, computed_forces, output_path, title: str | None = None) -> None:
    """Save a force scatter plot with all Cartesian components on one axis."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reference = np.asarray(reference_forces)
    computed = np.asarray(computed_forces)
    if reference.shape != computed.shape:
        raise ValueError("reference_forces and computed_forces must have the same shape.")
    if reference.ndim != 2 or reference.shape[1] != 3:
        raise ValueError("forces must have shape (N, 3).")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = ("Fx", "Fy", "Fz")
    colors = ("tab:blue", "tab:orange", "tab:green")
    all_values = np.concatenate((reference.reshape(-1), computed.reshape(-1)))
    vmin = float(np.min(all_values))
    vmax = float(np.max(all_values))
    if vmin == vmax:
        margin = abs(vmin) * 0.05 + 1.0
    else:
        margin = 0.05 * (vmax - vmin)
    lo = vmin - margin
    hi = vmax + margin

    fig, ax = plt.subplots(figsize=(7.0, 7.0), constrained_layout=True)
    for component, (label, color) in enumerate(zip(labels, colors)):
        ax.scatter(
            reference[:, component],
            computed[:, component],
            s=8,
            alpha=0.55,
            label=label,
            color=color,
            edgecolors="none",
        )
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.0, linestyle="--", label="y=x")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Direct reference force, N")
    ax.set_ylabel("FMM force, N")
    ax.grid(True, alpha=0.25)
    ax.legend()
    if title:
        ax.set_title(title)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
