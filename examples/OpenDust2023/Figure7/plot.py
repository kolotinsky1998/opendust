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

    _, history = load_case(results_dir, 0)
    time_us = history["time_s"] * 1.0e6
    orbit_pN = history["orbit_force_N"][:, 0, 2] * 1.0e12
    collection_pN = history["collection_force_N"][:, 0, 2] * 1.0e12
    total_pN = orbit_pN + collection_pN
    atomic_csv(
        results_dir / "aggregate.csv",
        np.column_stack((time_us, total_pN, orbit_pN, collection_pN)),
        "time_us,total_ion_force_z_pN,orbit_force_z_pN,collection_force_z_pN",
    )
    fig, ax = plt.subplots(figsize=(7.4, 4.6), constrained_layout=True)
    ax.plot(time_us, total_pN, color="#58c9cf", linewidth=0.9, label="Total")
    ax.plot(time_us, orbit_pN, color="#f04f3d", linewidth=1.2, label="Electrostatic interactions")
    ax.set_xlabel("Time (µs)")
    ax.set_ylabel("Force acting on the dust particle (pN)")
    ax.set_xlim(left=0.0)
    ax.grid(alpha=0.22)
    ax.legend()
    save_figure(fig, results_dir, "figure7_ion_drag_history")
    plt.close(fig)


if __name__ == "__main__":
    run_plot_cli(HERE, CASE_COUNT, generate)
