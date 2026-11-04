#!/usr/bin/env python3
"""Create figures, CSV tables, and validation_report.json from completed stages."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from opendust_jax import ARTICLE_CASES
from opendust_jax.article_reproduction import ARTICLE_REFERENCE_DENSITY_M3
from workflow import artifact_prefix, load_config, read_stage, smoke_config


HERE = Path(__file__).resolve().parent
ELEMENTARY_CHARGE_C = 1.602176634e-19
TABLE_I = {
    "linear": np.asarray(
        [[-4.297, 0.239, 0.0, 0.319], [-4.092, 0.206, 0.0, 0.322], [-4.299, 0.240, 0.0, 0.320]]
    ),
    "quadratic": np.asarray(
        [[0.646, -1.500, 0.150, 0.302], [-0.798, -1.001, 0.108, 0.300], [-0.720, -1.000, 0.106, 0.307]]
    ),
}
TABLE_II_CASE_I_CHARGE_1E4 = {
    "linear": np.asarray([1.448, 1.208, 1.162, 1.119, 1.117, 1.161, 1.176]),
    "quadratic": np.asarray([1.426, 1.242, 1.194, 1.144, 1.148, 1.128, 1.131]),
}


def stage(config, case_index: int, branch: str, iteration: int):
    prefix = artifact_prefix(config, case_index, branch, iteration)
    metadata, arrays = read_stage(prefix)
    if metadata.get("status") != "complete":
        raise RuntimeError(f"incomplete stage: {prefix}")
    return prefix, metadata, arrays


def write_csv(path: Path, header: list[str], rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def plot_figure3(config, output: Path) -> None:
    stages = [
        ("constant", 1, "constant"),
        ("linear", config.iterations, "linear, iteration 5"),
        ("quadratic", config.iterations, "quadratic, iteration 5"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    rows = []
    for branch, iteration, label in stages:
        _, _, data = stage(config, 1, branch, iteration)
        z = data["background_z_m"]
        density = np.mean(data["background_density_m3"], axis=1)
        velocity = np.divide(
            np.mean(data["background_axial_flux_m2_s"], axis=1),
            density,
            out=np.zeros_like(density),
            where=density > 0.0,
        )
        cs = np.sqrt(1.380649e-23 * 58_022.0 / 6.6335209e-26)
        lambda_de = (max(z) - min(z)) / config.height_screening_lengths
        z_over_lambda = z / lambda_de
        axes[0].plot(z_over_lambda, density / ARTICLE_REFERENCE_DENSITY_M3, label=label)
        axes[1].plot(z_over_lambda, velocity / cs, label=label)
        rows.extend(zip([label] * len(z), z_over_lambda, density / ARTICLE_REFERENCE_DENSITY_M3, velocity / cs))
    axes[0].set(xlabel=r"$z_{sim}/\lambda_{De}$", ylabel=r"$n_i/n_{i0}$")
    axes[1].set(xlabel=r"$z_{sim}/\lambda_{De}$", ylabel=r"$v_d/C_s$")
    axes[0].legend(fontsize=8)
    fig.savefig(output / "figure_3.png", dpi=220)
    fig.savefig(output / "figure_3.pdf")
    plt.close(fig)
    write_csv(output / "figure_3.csv", ["profile", "z_sim_over_lambda", "n_over_n0", "vz_over_cs"], rows)


def plot_iteration_figure(config, output: Path, branch: str, iteration: int, stem: str) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), constrained_layout=True)
    csv_rows = []
    for case_index, case in enumerate(ARTICLE_CASES):
        prefix, metadata, data = stage(config, case_index, branch, iteration)
        with np.load(prefix.with_name(prefix.name + "_fields.npz"), allow_pickle=False) as fields:
            x = np.asarray(fields["x_m"])
            z = np.asarray(fields["z_m"])
            density = np.asarray(fields["ion_density_m3"])
        image = axes[0, case_index].pcolormesh(
            x * 1e3, z * 1e3, density / ARTICLE_REFERENCE_DENSITY_M3, shading="auto"
        )
        fig.colorbar(image, ax=axes[0, case_index], label=r"$n_i/n_0$")
        axes[0, case_index].set_title(case.name.replace("_", " "))
        axes[0, case_index].set(xlabel="x [mm]", ylabel="z sim [mm]")
        z_paper = data["dust_z_paper_mm"]
        charge = -data["mean_charge_C"] / ELEMENTARY_CHARGE_C / 1e4
        charge_error = data["std_charge_C"] / ELEMENTARY_CHARGE_C / 1e4
        order = np.argsort(z_paper)
        axes[1, case_index].errorbar(z_paper[order], charge[order], yerr=charge_error[order], fmt="o-")
        axes[1, case_index].set(xlabel="z paper [mm]", ylabel=r"$-Q_d/(10^4e)$")
        required = -data["required_field_sim_V_m"] / 1e3
        imposed = -data["input_field_sim_V_m"] / 1e3
        axes[2, case_index].plot(z_paper[order], required[order], "o", label="force balance")
        axes[2, case_index].plot(z_paper[order], imposed[order], "-", label="imposed")
        axes[2, case_index].set(xlabel="z paper [mm]", ylabel="E paper [V/mm]")
        axes[2, case_index].legend(fontsize=8)
        for i in range(len(z_paper)):
            csv_rows.append(
                [case.name, i + 1, z_paper[i], charge[i], charge_error[i], required[i], imposed[i]]
            )
    fig.savefig(output / f"{stem}.png", dpi=220)
    fig.savefig(output / f"{stem}.pdf")
    plt.close(fig)
    write_csv(
        output / f"{stem}.csv",
        ["case", "dust", "z_paper_mm", "minus_charge_1e4e", "charge_std_1e4e", "required_E_V_mm", "imposed_E_V_mm"],
        csv_rows,
    )


def plot_figure6_7b(config, output: Path) -> None:
    plot_iteration_figure(config, output, "quadratic", config.iterations, "figure_6")
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    rows = []
    for iteration in range(2, config.iterations + 1):
        _, _, data = stage(config, 0, "quadratic", iteration)
        imbalance = 100.0 * np.abs(data["normalized_force_imbalance"])
        ax.plot(np.arange(1, 8), imbalance, "o-", label=f"iteration {iteration}")
        rows.extend((iteration, index + 1, value) for index, value in enumerate(imbalance))
    ax.set(xlabel="dust d1...d7", ylabel=r"$|F_T|/(m_dg)$ [%]", xticks=np.arange(1, 8))
    ax.legend(fontsize=8)
    fig.savefig(output / "figure_7b.png", dpi=220)
    fig.savefig(output / "figure_7b.pdf")
    plt.close(fig)
    write_csv(output / "figure_7b.csv", ["iteration", "dust", "absolute_force_imbalance_percent"], rows)
    left = plt.imread(output / "figure_6.png")
    right = plt.imread(output / "figure_7b.png")
    combined, combined_axes = plt.subplots(
        1, 2, figsize=(17, 8), gridspec_kw={"width_ratios": [1.7, 1.0]}, constrained_layout=True
    )
    combined_axes[0].imshow(left)
    combined_axes[0].set_title("Figure 6: quadratic iteration 5")
    combined_axes[1].imshow(right)
    combined_axes[1].set_title("Figure 7b: Case I force convergence")
    for axis in combined_axes:
        axis.axis("off")
    combined.savefig(output / "figure_6_7b_combined.png", dpi=220)
    combined.savefig(output / "figure_6_7b_combined.pdf")
    plt.close(combined)


def validation_report(config) -> dict:
    checks = {}
    final_values = {}
    for branch in ("linear", "quadratic"):
        values = []
        for case_index in range(3):
            _, metadata, _ = stage(config, case_index, branch, config.iterations)
            field = metadata["paper_field"]
            values.append(
                [field["E0_V_mm"], field["alpha_V_mm2"], field["beta_V_mm3"], metadata["mach"]]
            )
        values = np.asarray(values)
        final_values[branch] = values.tolist()
        reference = TABLE_I[branch]
        scale = np.maximum(np.abs(reference), 1.0e-12)
        relevant = np.ones_like(reference, dtype=bool)
        if branch == "linear":
            relevant[:, 2] = False
        error = np.abs(values - reference) / scale
        checks[f"table_I_{branch}_within_10_percent"] = bool(np.all(error[relevant] <= 0.10))
        checks[f"table_I_{branch}_max_relative_error"] = float(np.max(error[relevant]))

    charge_checks = {}
    rms = {}
    for branch in ("linear", "quadratic"):
        _, _, data = stage(config, 0, branch, config.iterations)
        charge = -data["mean_charge_C"] / ELEMENTARY_CHARGE_C / 1e4
        charge_error = np.abs(charge - TABLE_II_CASE_I_CHARGE_1E4[branch]) / TABLE_II_CASE_I_CHARGE_1E4[branch]
        charge_checks[branch] = {
            "computed_1e4e": charge.tolist(),
            "max_relative_error": float(np.max(charge_error)),
            "within_10_percent": bool(np.all(charge_error <= 0.10)),
        }
        rms[branch] = float(np.sqrt(np.mean(data["normalized_force_imbalance"] ** 2)))
    checks["quadratic_rms_force_below_5_percent"] = rms["quadratic"] <= 0.05
    checks["linear_residual_exceeds_quadratic"] = rms["linear"] > rms["quadratic"]
    checks["case_I_table_II_within_10_percent"] = all(
        item["within_10_percent"] for item in charge_checks.values()
    )
    convergence = {}
    for branch in ("linear", "quadratic"):
        convergence[branch] = []
        for case_index in range(3):
            _, metadata, _ = stage(config, case_index, branch, config.iterations)
            convergence[branch].append(
                {
                    "case": ARTICLE_CASES[case_index].name,
                    "relative_field_change": metadata["relative_field_change"],
                    "converged_at_one_percent": metadata["converged_at_one_percent"],
                }
            )
    required_boolean = [value for value in checks.values() if isinstance(value, bool)]
    return {
        "status": "pass" if all(required_boolean) else "needs_review",
        "checks": checks,
        "final_table_I_values": final_values,
        "case_I_charge_validation": charge_checks,
        "case_I_rms_normalized_force": rms,
        "iteration_convergence": convergence,
        "profile_trend_and_wake_peak_review": "manual review required from figures 3, 5, and 6",
        "known_model_difference": (
            "OpenDust physical absorption radius and orbit+collection force are used; "
            "the DRIAD b_c and Maxwell-stress construction are not reproduced."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=HERE / "article_config.json")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.smoke:
        config = smoke_config(config)
    output = Path(config.output_dir) / "figures"
    output.mkdir(parents=True, exist_ok=True)
    plot_figure3(config, output)
    plot_iteration_figure(config, output, "constant", 1, "figure_4")
    plot_iteration_figure(config, output, "linear", config.iterations, "figure_5")
    plot_figure6_7b(config, output)
    report = validation_report(config)
    (output / "validation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"wrote figures and validation report to {output}")


if __name__ == "__main__":
    main()
