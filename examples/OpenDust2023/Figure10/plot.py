#!/usr/bin/env python3
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from common import ELEMENTARY_CHARGE_C, atomic_csv, load_case, run_plot_cli, save_figure
from config import CASE_COUNT


EPSILON_0_F_M = 8.8541878128e-12


def generate(results_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for index in range(CASE_COUNT):
        summary, _ = load_case(results_dir, index)
        density = summary["plasma"]["density_m3"]
        ion_mass = summary["plasma"]["ion_mass_kg"]
        omega_pi = np.sqrt(
            density * ELEMENTARY_CHARGE_C**2 / (EPSILON_0_F_M * ion_mass)
        )
        orbit = summary["mean_orbit_force_N"][0][2]
        collection = summary["mean_collection_force_N"][0][2]
        rows.append(
            (
                summary["case"]["collision_frequency_s"] / omega_pi,
                (orbit + collection) * 1.0e12,
                summary["mean_charge_C"][0] / ELEMENTARY_CHARGE_C / 1.0e4,
                orbit * 1.0e12,
                collection * 1.0e12,
            )
        )
    table = np.asarray(sorted(rows), dtype=np.float64)
    atomic_csv(
        results_dir / "aggregate.csv",
        table,
        "collision_frequency_over_omega_pi,total_ion_force_z_pN,charge_1e4_e,orbit_force_z_pN,collection_force_z_pN",
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.3), constrained_layout=True)
    axes[0].plot(table[:, 0], table[:, 1], "o-", color="#e85c4a", linewidth=1.5, label="OpenDustJAX")
    axes[0].set_xscale("log")
    axes[0].set_xlabel(r"$\nu_{in}/\omega_{pi}$")
    axes[0].set_ylabel("Force acting on the dust particle (pN)")
    axes[0].grid(alpha=0.22, which="both")
    axes[0].legend()
    axes[1].plot(table[:, 0], table[:, 2], "o-", color="#e85c4a", linewidth=1.5, label="OpenDustJAX")
    axes[1].set_xscale("log")
    axes[1].set_xlabel(r"$\nu_{in}/\omega_{pi}$")
    axes[1].set_ylabel(r"Dust particle charge ($10^4 e$)")
    axes[1].grid(alpha=0.22, which="both")
    axes[1].legend()
    save_figure(fig, results_dir, "figure10_collisional_flow")
    plt.close(fig)


if __name__ == "__main__":
    run_plot_cli(HERE, CASE_COUNT, generate)
