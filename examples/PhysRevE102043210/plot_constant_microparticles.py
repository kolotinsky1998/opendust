#!/usr/bin/env python3
"""Plot microparticle diagnostics for completed constant-field iterations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ELEMENTARY_CHARGE_C = 1.602176634e-19
CASE_COLORS = {
    "case_I": "#0072B2",
    "case_II": "#D55E00",
    "case_III": "#009E73",
}


def load_completed_constant_iterations(results: Path) -> list[tuple[dict, dict[str, np.ndarray], Path]]:
    stages = []
    for metadata_path in sorted(results.glob("case_*/constant/iteration_[0-9][0-9].json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        data_path = metadata_path.with_suffix(".npz")
        if metadata.get("status") != "complete" or not data_path.exists():
            continue
        with np.load(data_path, allow_pickle=False) as archive:
            arrays = {key: np.asarray(archive[key]) for key in archive.files}
        stages.append((metadata, arrays, data_path))
    return stages


def stage_label(metadata: dict) -> str:
    case = metadata["case"]["name"].replace("case_", "Case ")
    return f"{case}, iteration {metadata['iteration']}"


def setup_particle_axis(ax: plt.Axes, count: int) -> np.ndarray:
    particle = np.arange(1, count + 1)
    ax.set_xticks(particle)
    ax.set_xlabel("Microparticle index (1 = uppermost)")
    ax.grid(alpha=0.22)
    return particle


def plot_particle_summary(stages, output: Path) -> None:
    fig, axes = plt.subplots(
        len(stages), 4, figsize=(17, 4.2 * len(stages)), squeeze=False,
        constrained_layout=True,
    )
    for row, (metadata, data, _) in enumerate(stages):
        label = stage_label(metadata)
        color = CASE_COLORS.get(metadata["case"]["name"], f"C{row}")
        count = len(data["mean_charge_C"])
        particle = np.arange(1, count + 1)

        ax = axes[row, 0]
        charge = -data["mean_charge_C"] / ELEMENTARY_CHARGE_C / 1.0e4
        charge_std = data["std_charge_C"] / ELEMENTARY_CHARGE_C / 1.0e4
        ax.errorbar(particle, charge, yerr=charge_std, color=color, marker="o", capsize=3)
        ax.set_ylabel(r"Charge magnitude $-Q/(10^4 e)$")
        ax.set_title(f"{label}: charge")
        setup_particle_axis(ax, count)

        ax = axes[row, 1]
        force_series = (
            ("Orbit", data["mean_orbit_z_N"], "o", "-"),
            ("Collection", data["mean_collection_z_N"], "s", "--"),
            ("Dust–dust", data["mean_dust_dust_z_N"], "^", "-."),
        )
        total = np.zeros(count)
        for component, values, marker, line_style in force_series:
            values_pN = values * 1.0e12
            total += values_pN
            ax.plot(particle, values_pN, marker=marker, ls=line_style, label=component)
        ax.plot(particle, total, color="black", marker="D", lw=2.0, label="Sum")
        ax.axhline(0.0, color="0.55", lw=0.8)
        ax.set_ylabel(r"Axial force $F_z$ (pN)")
        ax.set_title("Time-averaged force components")
        setup_particle_axis(ax, count)
        ax.legend(frameon=False, fontsize=8, ncol=2)

        ax = axes[row, 2]
        required_paper = -data["required_field_sim_V_m"] / 1.0e3
        imposed_paper = -data["input_field_sim_V_m"] / 1.0e3
        ax.plot(particle, required_paper, color=color, marker="o", label="Required by force balance")
        ax.plot(particle, imposed_paper, color="0.25", ls="--", label="Imposed constant field")
        ax.set_ylabel(r"Paper-coordinate field $E_z$ (V/mm)")
        ax.set_title("Field required for equilibrium")
        setup_particle_axis(ax, count)
        ax.legend(frameon=False, fontsize=8)

        ax = axes[row, 3]
        imbalance = 100.0 * data["normalized_force_imbalance"]
        ax.plot(particle, imbalance, color=color, marker="o")
        ax.axhline(0.0, color="black", lw=0.9)
        ax.set_ylabel(r"Net force / weight (%)")
        ax.set_title("Residual force imbalance")
        setup_particle_axis(ax, count)

        z_text = ", ".join(f"{z:.2f}" for z in data["dust_z_paper_mm"])
        axes[row, 0].text(
            0.02, -0.24, rf"$z_{{paper}}$ (mm), particles 1–{count}: {z_text}",
            transform=axes[row, 0].transAxes, fontsize=8, color="0.35",
        )

    fig.suptitle("Completed constant-field iterations: microparticle diagnostics", fontsize=16)
    fig.savefig(output / "constant_microparticle_summary.png", dpi=220)
    fig.savefig(output / "constant_microparticle_summary.pdf")
    plt.close(fig)


def block_average(values: np.ndarray, block_size: int) -> np.ndarray:
    block_count = len(values) // block_size
    if block_count == 0:
        return values
    used = block_count * block_size
    return values[:used].reshape(block_count, block_size, *values.shape[1:]).mean(axis=1)


def plot_time_histories(stages, output: Path, block_size: int = 50) -> None:
    fig, axes = plt.subplots(
        len(stages), 2, figsize=(13, 4.2 * len(stages)), squeeze=False,
        constrained_layout=True,
    )
    particle_colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, 7))
    for row, (metadata, data, _) in enumerate(stages):
        period = float(metadata["ion_plasma_period_s"])
        burn = int(metadata["burn_in_step"])
        time_periods = data["times_s"] / period
        burn_time = data["times_s"][burn] / period

        charge = -data["charge_history_C"] / ELEMENTARY_CHARGE_C / 1.0e4
        other_force = (
            data["orbit_force_history_N"][:, :, 2]
            + data["collection_force_history_N"][:, :, 2]
            + data["dust_dust_force_history_N"][:, :, 2]
        ) * 1.0e12
        block_time = block_average(time_periods[:, None], block_size)[:, 0]
        block_charge = block_average(charge, block_size)
        block_force = block_average(other_force, block_size)

        for particle in range(charge.shape[1]):
            line_label = f"Particle {particle + 1}" if row == 0 else None
            axes[row, 0].plot(
                block_time, block_charge[:, particle], color=particle_colors[particle],
                lw=1.2, label=line_label,
            )
            axes[row, 1].plot(
                block_time, block_force[:, particle], color=particle_colors[particle],
                lw=1.2, label=line_label,
            )

        for ax in axes[row]:
            ax.axvline(burn_time, color="tab:red", ls="--", lw=1.1, label="Burn-in end" if row == 0 else None)
            ax.set_xlabel(r"Time $t/\tau_i$")
            ax.grid(alpha=0.2)
        axes[row, 0].set_ylabel(r"Charge magnitude $-Q/(10^4 e)$")
        axes[row, 1].set_ylabel(r"Non-electric axial force (pN)")
        axes[row, 0].set_title(f"{stage_label(metadata)}: block-averaged charge")
        axes[row, 1].set_title(f"{stage_label(metadata)}: block-averaged force")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=8, frameon=False)
    fig.savefig(output / "constant_microparticle_histories.png", dpi=220)
    fig.savefig(output / "constant_microparticle_histories.pdf")
    plt.close(fig)


def write_summary_csv(stages, output: Path) -> None:
    columns = [
        "case", "iteration", "particle", "z_paper_mm", "charge_C", "charge_1e4_e",
        "orbit_force_z_pN", "collection_force_z_pN", "dust_dust_force_z_pN",
        "required_field_paper_V_mm", "imposed_field_paper_V_mm", "force_imbalance_percent",
    ]
    with (output / "constant_microparticle_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for metadata, data, _ in stages:
            for particle in range(len(data["mean_charge_C"])):
                writer.writerow({
                    "case": metadata["case"]["name"],
                    "iteration": metadata["iteration"],
                    "particle": particle + 1,
                    "z_paper_mm": data["dust_z_paper_mm"][particle],
                    "charge_C": data["mean_charge_C"][particle],
                    "charge_1e4_e": -data["mean_charge_C"][particle] / ELEMENTARY_CHARGE_C / 1.0e4,
                    "orbit_force_z_pN": data["mean_orbit_z_N"][particle] * 1.0e12,
                    "collection_force_z_pN": data["mean_collection_z_N"][particle] * 1.0e12,
                    "dust_dust_force_z_pN": data["mean_dust_dust_z_N"][particle] * 1.0e12,
                    "required_field_paper_V_mm": -data["required_field_sim_V_m"][particle] / 1.0e3,
                    "imposed_field_paper_V_mm": -data["input_field_sim_V_m"][particle] / 1.0e3,
                    "force_imbalance_percent": 100.0 * data["normalized_force_imbalance"][particle],
                })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path, help="Root of downloaded workflow results")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--history-block-size", type=int, default=50)
    args = parser.parse_args()
    if args.history_block_size < 1:
        raise ValueError("--history-block-size must be positive")

    stages = load_completed_constant_iterations(args.results)
    if not stages:
        raise RuntimeError(f"no completed constant-field iterations found under {args.results}")
    args.output.mkdir(parents=True, exist_ok=True)
    plot_particle_summary(stages, args.output)
    plot_time_histories(stages, args.output, args.history_block_size)
    write_summary_csv(stages, args.output)
    print(f"plotted {len(stages)} completed constant-field iterations in {args.output}")


if __name__ == "__main__":
    main()
