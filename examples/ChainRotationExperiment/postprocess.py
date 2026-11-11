#!/usr/bin/env python3
"""Create figures and CSV tables for the chain-rotation parameter grid."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
import math
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from opendust_jax.correlation_analysis import normalized_autocorrelation  # noqa: E402
from opendust_jax.models import ELEMENTARY_CHARGE_C  # noqa: E402
from workflow import (  # noqa: E402
    artifact_prefix,
    build_geometry,
    cases,
    load_config,
    read_stage,
)


def write_csv(path: Path, header: list[str], rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def save_figure(figure: plt.Figure, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    figure.savefig(directory / f"{stem}.png", dpi=220, bbox_inches="tight")
    figure.savefig(directory / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)


def completed_stage(config, case, iteration: int):
    prefix = artifact_prefix(config, case, iteration)
    metadata, arrays = read_stage(prefix)
    if metadata.get("status") != "complete":
        raise RuntimeError(f"incomplete stage: {prefix}")
    return prefix, metadata, arrays


def diagnostic_fields(prefix: Path):
    path = prefix.with_name(prefix.name + "_fields.npz")
    with np.load(path, allow_pickle=False) as fields:
        if "plane_labels" in fields:
            labels = tuple(str(value) for value in fields["plane_labels"])
            transverse = np.asarray(fields["transverse_m"])
            potential = np.asarray(fields["potential_V"])
            density = np.asarray(fields["ion_density_m3"])
        else:
            labels = ("xz",)
            transverse = np.asarray(fields["x_m"])
            potential = np.asarray(fields["potential_V"])[None, ...]
            density = np.asarray(fields["ion_density_m3"])[None, ...]
        return {
            "labels": labels,
            "transverse_m": transverse,
            "z_m": np.asarray(fields["z_m"]),
            "potential_V": potential,
            "density_m3": density,
        }


def plot_geometry(config, output: Path) -> None:
    case = cases(config)[0]
    geometry = build_geometry(config, case)
    laboratory = geometry.laboratory_positions_m * 1.0e3
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    specifications = ((0, 1, "x", "y"), (0, 2, "x", "z"), (1, 2, "y", "z"))
    for axis, (left, right, left_name, right_name) in zip(axes, specifications):
        axis.plot(laboratory[:, left], laboratory[:, right], "o-")
        for particle_id, point in zip(geometry.particle_ids, laboratory):
            axis.annotate(str(particle_id), (point[left], point[right]), xytext=(4, 4), textcoords="offset points")
        axis.set(xlabel=f"{left_name} [mm]", ylabel=f"{right_name} [mm]")
        axis.grid(alpha=0.25)
        axis.set_aspect("equal", adjustable="datalim")
    save_figure(figure, output, "chain_geometry")
    write_csv(
        output / "chain_geometry.csv",
        ["particle_id", "x_m", "y_m", "z_m"],
        zip(geometry.particle_ids, *geometry.laboratory_positions_m.T),
    )


def plot_maps(config, case, output: Path) -> None:
    selections = ((1, "constant"), (config.iterations, "quadratic final"))
    figure, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    rows = []
    for row_index, (iteration, stage_label) in enumerate(selections):
        prefix, _, arrays = completed_stage(config, case, iteration)
        fields = diagnostic_fields(prefix)
        positions = arrays["simulation_positions_m"] * 1.0e3
        for plane_index, plane in enumerate(("xz", "yz")):
            source_index = fields["labels"].index(plane)
            transverse = fields["transverse_m"] * 1.0e3
            z = fields["z_m"] * 1.0e3
            density = fields["density_m3"][source_index] / case.reference_density_m3
            potential = fields["potential_V"][source_index]
            coordinate_index = 0 if plane == "xz" else 1
            density_axis = axes[row_index, 2 * plane_index]
            potential_axis = axes[row_index, 2 * plane_index + 1]
            density_image = density_axis.pcolormesh(transverse, z, density, shading="auto")
            potential_image = potential_axis.pcolormesh(transverse, z, potential, shading="auto")
            for axis in (density_axis, potential_axis):
                axis.plot(positions[:, coordinate_index], positions[:, 2], "wo", ms=3, mec="k", mew=0.4)
                axis.set(xlabel=f"{plane[0]} [mm]", ylabel="z [mm]")
            density_axis.set_title(f"{stage_label}, {plane.upper()}: $n_i/n_0$")
            potential_axis.set_title(f"{stage_label}, {plane.upper()}: potential [V]")
            figure.colorbar(density_image, ax=density_axis)
            figure.colorbar(potential_image, ax=potential_axis)
            for iz, z_value in enumerate(fields["z_m"]):
                for it, transverse_value in enumerate(fields["transverse_m"]):
                    rows.append(
                        (
                            iteration,
                            plane,
                            transverse_value,
                            z_value,
                            density[iz, it],
                            potential[iz, it],
                        )
                    )
    save_figure(figure, output, "density_potential_maps")
    write_csv(
        output / "density_potential_maps.csv",
        ["iteration", "plane", "transverse_m", "z_m", "density_over_n0", "potential_V"],
        rows,
    )


def plot_profiles(config, case, output: Path) -> None:
    prefix, metadata, arrays = completed_stage(config, case, config.iterations)
    fields = diagnostic_fields(prefix)
    z_background = arrays["background_z_m"]
    density_background = np.mean(arrays["background_density_m3"], axis=1)
    velocity_background = np.divide(
        np.mean(arrays["background_axial_flux_m2_s"], axis=1),
        density_background,
        out=np.zeros_like(density_background),
        where=density_background > 0.0,
    )
    sound_speed = math.sqrt(
        1.380649e-23 * case.electron_temperature_K / config.ion_mass_kg
    )
    central_profiles = {}
    for plane_index, plane in enumerate(fields["labels"]):
        central_profiles[plane] = np.asarray(
            [
                np.interp(0.0, fields["transverse_m"], row)
                for row in fields["density_m3"][plane_index]
            ]
        )
    mean_central = np.mean(np.stack(list(central_profiles.values())), axis=0)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot(z_background * 1e3, density_background / case.reference_density_m3)
    axes[0].set(xlabel="z [mm]", ylabel="$n_i/n_0$", title="Presheath radial mean")
    axes[1].plot(fields["z_m"] * 1e3, mean_central / case.reference_density_m3, label="XZ/YZ mean")
    for plane, values in central_profiles.items():
        axes[1].plot(fields["z_m"] * 1e3, values / case.reference_density_m3, "--", label=plane.upper())
    axes[1].set(xlabel="z [mm]", ylabel="$n_i(0,z)/n_0$", title="Central-axis profile")
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.25)
    save_figure(figure, output, "density_profiles")
    write_csv(
        output / "background_profile.csv",
        ["z_m", "density_m3", "density_over_n0", "axial_velocity_m_s", "mach"],
        zip(
            z_background,
            density_background,
            density_background / case.reference_density_m3,
            velocity_background,
            velocity_background / sound_speed,
        ),
    )
    rows = []
    for index, z_value in enumerate(fields["z_m"]):
        rows.append(
            [
                z_value,
                *(central_profiles[plane][index] for plane in fields["labels"]),
                mean_central[index],
            ]
        )
    write_csv(
        output / "central_axis_density_profile.csv",
        ["z_m", *(f"density_{plane}_m3" for plane in fields["labels"]), "density_plane_mean_m3"],
        rows,
    )


def plot_charge_and_forces(config, case, output: Path) -> None:
    stages = [completed_stage(config, case, iteration) for iteration in range(1, 6)]
    particle_ids = stages[0][2]["particle_ids"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    charge_rows = []
    for iteration, (_, _, arrays) in enumerate(stages, start=1):
        charge_e = -arrays["mean_charge_C"] / ELEMENTARY_CHARGE_C
        charge_ci = 1.96 * arrays["sem_charge_C"] / ELEMENTARY_CHARGE_C
        axes[0].errorbar(particle_ids, charge_e, yerr=charge_ci, marker="o", label=f"iteration {iteration}")
        axes[1].plot(particle_ids, 100.0 * arrays["relative_charge_sem"], "o-", label=f"iteration {iteration}")
        for index, particle_id in enumerate(particle_ids):
            charge_rows.append(
                (
                    iteration,
                    particle_id,
                    arrays["mean_charge_C"][index],
                    arrays["sem_charge_C"][index],
                    arrays["relative_charge_sem"][index],
                    arrays["tau_int_s_charge_C"][index],
                    arrays["effective_sample_size_charge_C"][index],
                )
            )
    axes[0].set(xlabel="particle id", ylabel="$-Q/e$", title="Charge with correlation-corrected 95% CI")
    axes[1].axhline(1.0, color="k", lw=0.8, ls="--")
    axes[1].set(xlabel="particle id", ylabel="relative SEM [%]", title="Charge mean uncertainty")
    axes[0].legend(fontsize=7)
    for axis in axes:
        axis.grid(alpha=0.25)
    save_figure(figure, output, "charge_iterations")
    write_csv(
        output / "charge_iterations.csv",
        ["iteration", "particle_id", "mean_charge_C", "sem_charge_C", "relative_sem", "tau_int_s", "effective_sample_size"],
        charge_rows,
    )

    figure, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    force_rows = []
    for iteration, (_, _, arrays) in enumerate(stages, start=1):
        mean = arrays["mean_total_force_N"] * 1e12
        sem = arrays["sem_total_force_N"] * 1e12
        for component, axis in enumerate(axes):
            axis.errorbar(particle_ids, mean[:, component], yerr=1.96 * sem[:, component], marker="o", label=f"iteration {iteration}")
        for particle_index, particle_id in enumerate(particle_ids):
            for component, component_name in enumerate("xyz"):
                force_rows.append(
                    (
                        iteration,
                        particle_id,
                        component_name,
                        arrays["mean_total_force_N"][particle_index, component],
                        arrays["sem_total_force_N"][particle_index, component],
                        arrays["tau_int_s_total_force_N"][particle_index, component],
                        arrays["effective_sample_size_total_force_N"][particle_index, component],
                    )
                )
    for component, axis in zip("xyz", axes):
        axis.axhline(0.0, color="k", lw=0.8)
        axis.set(xlabel="particle id", ylabel=f"$F_{component}$ [pN]", title=f"Total $F_{component}$")
        axis.grid(alpha=0.25)
    axes[-1].legend(fontsize=7)
    save_figure(figure, output, "total_force_iterations")
    write_csv(
        output / "total_force_iterations.csv",
        ["iteration", "particle_id", "component", "mean_force_N", "sem_force_N", "tau_int_s", "effective_sample_size"],
        force_rows,
    )

    final = stages[-1][2]
    mechanisms = (
        "orbit",
        "collection",
        "dust_dust",
        "external",
        "gravity",
        "total",
    )
    figure, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    decomposition_rows = []
    for component, axis in enumerate(axes):
        for mechanism in mechanisms:
            values = final[f"mean_{mechanism}_force_N"][:, component] * 1e12
            axis.plot(particle_ids, values, "o-", label=mechanism.replace("_", "-"))
            decomposition_rows.extend(
                (particle_id, "xyz"[component], mechanism, value / 1e12)
                for particle_id, value in zip(particle_ids, values)
            )
        axis.axhline(0.0, color="k", lw=0.8)
        axis.set(xlabel="particle id", ylabel=f"$F_{'xyz'[component]}$ [pN]")
        axis.grid(alpha=0.25)
    axes[-1].legend(fontsize=7)
    save_figure(figure, output, "final_force_decomposition")
    write_csv(
        output / "final_force_decomposition.csv",
        ["particle_id", "component", "mechanism", "mean_force_N"],
        decomposition_rows,
    )


def plot_field_iterations(config, case, output: Path) -> None:
    figure, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    rows = []
    coefficient_rows = []
    for iteration in range(1, 6):
        _, metadata, arrays = completed_stage(config, case, iteration)
        z_mm = arrays["simulation_positions_m"][:, 2] * 1e3
        order = np.argsort(z_mm)
        required = arrays["required_field_sim_V_m"] / 1e3
        imposed = arrays["input_field_sim_V_m"] / 1e3
        ax.plot(z_mm[order], required[order], "o", label=f"required {iteration}")
        field = metadata["field"]
        dense_z_m = np.linspace(np.min(z_mm), np.max(z_mm), 300) * 1e-3
        dense_field_V_mm = (
            field["E0_V_m"]
            + field["alpha_V_m2"] * dense_z_m
            + field["beta_V_m3"] * dense_z_m**2
        ) / 1e3
        ax.plot(dense_z_m * 1e3, dense_field_V_mm, "-", alpha=0.65)
        for particle_id, z_value, required_value, imposed_value in zip(
            arrays["particle_ids"], z_mm, required, imposed
        ):
            rows.append((iteration, particle_id, z_value, required_value, imposed_value))
        coefficient_rows.append(
            (
                iteration,
                metadata["mach"],
                field["E0_V_m"],
                field["alpha_V_m2"],
                field["beta_V_m3"],
                metadata["relative_field_change"],
            )
        )
    ax.set(xlabel="z [mm]", ylabel="$E_z$ [V/mm]", title="Required points and imposed fields")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    save_figure(figure, output, "field_iterations")
    write_csv(
        output / "field_iterations.csv",
        ["iteration", "particle_id", "z_mm", "required_field_V_mm", "imposed_field_V_mm"],
        rows,
    )
    write_csv(
        output / "field_coefficients.csv",
        ["iteration", "mach", "E0_V_m", "alpha_V_m2", "beta_V_m3", "relative_field_change"],
        coefficient_rows,
    )


def plot_final_autocorrelations(config, case, output: Path) -> None:
    _, metadata, arrays = completed_stage(config, case, config.iterations)
    burn_in_step = int(metadata["burn_in_step"])
    total = arrays["total_force_history_N"][burn_in_step:]
    selected = {
        "charge_C": arrays["charge_history_C"][burn_in_step:],
        "total Fx": total[:, :, 0],
        "total Fy": total[:, :, 1],
        "total Fz": total[:, :, 2],
    }
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    dt_over_period = metadata["dt_s"] / metadata["ion_plasma_period_s"]
    for axis, (name, history) in zip(axes.flat, selected.items()):
        maximum_lag = min(history.shape[0] - 1, int(math.ceil(20.0 / dt_over_period)))
        lags = np.arange(maximum_lag + 1) * dt_over_period
        for particle_index in range(history.shape[1]):
            rho = normalized_autocorrelation(history[:, particle_index])
            axis.plot(lags, rho[: maximum_lag + 1], lw=0.8, label=f"p{particle_index + 1}")
        axis.axhline(0.0, color="k", lw=0.7)
        axis.axhline(math.exp(-1.0), color="0.5", lw=0.7, ls="--")
        axis.set(xlabel="lag [$\\tau_i$]", ylabel="ACF", title=name)
        axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=6, ncol=2)
    save_figure(figure, output, "final_autocorrelations")


def write_grid_summary(config, selected_cases, output: Path) -> None:
    rows = []
    for case in selected_cases:
        _, metadata, arrays = completed_stage(config, case, config.iterations)
        field = metadata["field"]
        rows.append(
            (
                case.index,
                case.name,
                case.electron_temperature_eV,
                case.reference_density_m3,
                metadata["screening_length_m"],
                metadata["radius_m"],
                metadata["height_m"],
                metadata["particle_count"],
                metadata["real_ion_count"],
                metadata["superion_weight"],
                metadata["superion_charge_C"],
                metadata["superion_to_initial_dust_charge_ratio"],
                metadata["dt_s"],
                metadata["steps"],
                metadata["mach"],
                field["E0_V_m"],
                field["alpha_V_m2"],
                field["beta_V_m3"],
                metadata["relative_field_change"],
                float(np.max(arrays["relative_charge_sem"])),
                metadata["charge_sem_above_one_percent"],
                metadata["stationarity_warning"],
            )
        )
    write_csv(
        output / "grid_summary.csv",
        [
            "case_index", "case", "electron_temperature_eV", "density_m3",
            "lambda_De_m", "radius_m", "height_m", "particle_count",
            "real_ion_count", "superion_weight", "superion_charge_C",
            "superion_to_initial_dust_charge_ratio", "dt_s", "steps", "final_mach",
            "final_E0_V_m", "final_alpha_V_m2", "final_beta_V_m3",
            "relative_field_change", "maximum_relative_charge_sem",
            "charge_sem_above_one_percent", "stationarity_warning",
        ],
        rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "chain_config.json")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.output_dir is not None:
        config = replace(config, output_dir=args.output_dir.resolve())
    if args.smoke:
        config = replace(config, output_dir=config.output_dir / "smoke")
    output = config.output_dir / "figures"
    selected_cases = cases(config)
    if args.smoke:
        selected_cases = tuple(
            case
            for case in selected_cases
            if artifact_prefix(config, case, config.iterations)
            .with_suffix(".json")
            .is_file()
        )
        if not selected_cases:
            raise FileNotFoundError("no completed smoke cases were found")
    plot_geometry(config, output)
    for case in selected_cases:
        case_output = output / case.name
        plot_maps(config, case, case_output)
        plot_profiles(config, case, case_output)
        plot_charge_and_forces(config, case, case_output)
        plot_field_iterations(config, case, case_output)
        plot_final_autocorrelations(config, case, case_output)
    write_grid_summary(config, selected_cases, output)
    (output / "report.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "cases": len(selected_cases),
                "iterations_per_case": config.iterations,
                "diagnostic_planes": list(config.diagnostic_planes),
                "output_directory": str(output),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"wrote chain-rotation figures and tables to {output}")


if __name__ == "__main__":
    main()
