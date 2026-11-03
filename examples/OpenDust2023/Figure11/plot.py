#!/usr/bin/env python3
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from common import atomic_csv, load_case, run_plot_cli, save_figure
from config import CASE_COUNT


def generate(results_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for index in range(CASE_COUNT):
        summary, _ = load_case(results_dir, index)
        orbit_x = summary["mean_orbit_force_N"][1][0]
        dust_x = summary["mean_dust_force_N"][1][0]
        rows.append(
            (
                summary["case"]["mach"],
                abs(orbit_x + dust_x) * 1.0e12,
                orbit_x * 1.0e12,
                dust_x * 1.0e12,
            )
        )
    table = np.asarray(sorted(rows), dtype=np.float64)
    atomic_csv(
        results_dir / "aggregate.csv",
        table,
        "mach,restoring_force_magnitude_pN,orbit_force_x_pN,dust_force_x_pN",
    )
    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    ax.plot(table[:, 0], table[:, 1], "o-", color="#e85c4a", linewidth=1.6, markersize=6.5, label="OpenDustJAX")
    ax.set_xlabel("Mach number")
    ax.set_ylabel("Restoring force (pN)")
    ax.set_xlim(0.25, 1.05)
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.22)
    ax.legend()
    save_figure(fig, results_dir, "figure11_restoring_force")
    plt.close(fig)


if __name__ == "__main__":
    run_plot_cli(HERE, CASE_COUNT, generate)
