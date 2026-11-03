#!/usr/bin/env python3
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from common import ELEMENTARY_CHARGE_C, atomic_csv, load_case, run_plot_cli, save_figure
from config import CASE_COUNT


def generate(results_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = {}
    times_us = None
    for index in range(CASE_COUNT):
        summary, history = load_case(results_dir, index)
        mode = summary["case"]["charge_mode"]
        times_us = history["time_s"] * 1.0e6
        series[mode] = np.abs(history["charge_C"][:, 0]) / ELEMENTARY_CHARGE_C / 1.0e4

    table = np.column_stack(
        (times_us, series["oml"], series["self_consistent"])
    )
    atomic_csv(
        results_dir / "aggregate.csv",
        table,
        "time_us,oml_charge_1e4_e,self_consistent_charge_1e4_e",
    )
    fig, ax = plt.subplots(figsize=(7.4, 4.6), constrained_layout=True)
    ax.plot(times_us, series["oml"], color="#58c9cf", linewidth=2.3, label="OML charging model")
    ax.plot(
        times_us,
        series["self_consistent"],
        color="#e95b4f",
        linewidth=1.2,
        label="Self-consistent ion flux",
    )
    ax.set_xlabel("Time (µs)")
    ax.set_ylabel(r"Dust particle charge magnitude ($10^4 e$)")
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.22)
    ax.legend()
    save_figure(fig, results_dir, "figure6_charge")
    plt.close(fig)


if __name__ == "__main__":
    run_plot_cli(HERE, CASE_COUNT, generate)
