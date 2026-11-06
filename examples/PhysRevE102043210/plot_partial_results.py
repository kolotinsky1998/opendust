#!/usr/bin/env python3
"""Visualize every completed article-reproduction stage, including failed next fits."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ELEMENTARY_CHARGE_C = 1.602176634e-19
REFERENCE_DENSITY_M3 = 1.0e14
ELECTRON_TEMPERATURE_K = 58_022.0
ION_MASS_KG = 6.6335209e-26
KB = 1.380649e-23
EPSILON_0 = 8.8541878128e-12
ARTICLE_LINEAR = {
    "case_I": (-4.297, 0.239),
    "case_II": (-4.092, 0.206),
    "case_III": (-4.299, 0.240),
}
ARTICLE_QUADRATIC = {
    "case_I": (0.646, -1.500, 0.150),
    "case_II": (-0.798, -1.001, 0.108),
    "case_III": (-0.720, -1.000, 0.106),
}
CASE_STYLE = {
    "case_I": ("#00b95a", "o", "Case 1"),
    "case_II": ("#e43d52", "^", "Case 2"),
    "case_III": ("#5056b8", "*", "Case 3"),
}


def stage_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.glob("case_*/*/iteration_*.npz")
        if not any(tag in path.stem for tag in ("_fields", "_restart", "_background"))
    )


def load_stage(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    return metadata, arrays


def paper_polynomial(coefficients: dict, z_mm: np.ndarray) -> np.ndarray:
    return (
        coefficients["E0_V_mm"]
        + coefficients["alpha_V_mm2"] * z_mm
        + coefficients["beta_V_mm3"] * z_mm**2
    )


def lambda_de_m() -> float:
    return math.sqrt(
        EPSILON_0 * KB * ELECTRON_TEMPERATURE_K
        / ((ELEMENTARY_CHARGE_C**2) * REFERENCE_DENSITY_M3)
    )


def latest(stages: dict, case: str, branch: str):
    choices = [key for key in stages if key[0] == case and key[1] == branch]
    return stages[max(choices, key=lambda item: item[2])] if choices else None


def candidate_status(case: str, branch: str, coefficients: dict, dust_z_paper_mm: np.ndarray):
    alpha = coefficients["alpha_V_mm2"]
    beta = coefficients["beta_V_mm3"]
    if branch == "linear":
        if alpha < 0.0:
            return False, f"invalid: slope = {alpha:.3f} < 0"
        return True, f"admissible: slope = {alpha:.3f}"
    if beta < 0.0:
        return False, f"invalid after coordinate transform: beta = {beta:.3f} < 0"
    vertex = -alpha / (2.0 * beta) if beta != 0.0 else math.inf
    lower, upper = np.min(dust_z_paper_mm), np.max(dust_z_paper_mm)
    if not lower < vertex < upper:
        return False, f"invalid: vertex {vertex:.2f} mm outside [{lower:.2f}, {upper:.2f}]"
    return True, f"admissible: vertex = {vertex:.2f} mm"


def plot_failed_fits(stages: dict, output: Path) -> None:
    cases = ("case_I", "case_II", "case_III")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True, sharex="col")
    for column, case in enumerate(cases):
        for row, branch in enumerate(("linear", "quadratic")):
            ax = axes[row, column]
            source = latest(stages, case, branch)
            if source is None:
                source = latest(stages, case, "constant")
            metadata, data, path = source
            fit = metadata.get("next_unweighted_fits", {}).get(branch)
            z_dust = np.asarray(data["dust_z_paper_mm"])
            order = np.argsort(z_dust)
            required = -np.asarray(data["required_field_sim_V_m"]) / 1.0e3
            imposed = -np.asarray(data["input_field_sim_V_m"]) / 1.0e3
            grid = np.linspace(z_dust.min() - 0.3, z_dust.max() + 0.3, 300)
            ax.plot(z_dust[order], imposed[order], "o--", color="0.5", label="imposed")
            ax.plot(z_dust[order], required[order], "ko", label="force balance")
            if fit is not None:
                ok, reason = candidate_status(case, branch, fit, z_dust)
                color = "tab:green" if ok else "tab:red"
                ax.plot(grid, paper_polynomial(fit, grid), color=color, lw=2.4, label="next raw fit")
                if branch == "quadratic" and fit["beta_V_mm3"] != 0.0:
                    vertex = -fit["alpha_V_mm2"] / (2.0 * fit["beta_V_mm3"])
                    ax.axvline(vertex, color=color, ls=":", lw=1.5)
                ax.text(
                    0.03, 0.04, reason, transform=ax.transAxes, color=color,
                    fontsize=9, bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": color},
                )
            reference = ARTICLE_LINEAR[case] if branch == "linear" else ARTICLE_QUADRATIC[case]
            reference_coefficients = {
                "E0_V_mm": reference[0],
                "alpha_V_mm2": reference[1],
                "beta_V_mm3": 0.0 if branch == "linear" else reference[2],
            }
            ax.plot(grid, paper_polynomial(reference_coefficients, grid), "b-.", lw=1.2, label="article fit")
            ax.axvspan(z_dust.min(), z_dust.max(), color="tab:blue", alpha=0.05)
            iteration = int(path.stem.split("_")[-1])
            ax.set_title(f"{case.replace('_', ' ').title()} — {branch}\nsource iteration {iteration}")
            ax.set_xlabel(r"$z_{paper}$ [mm]")
            ax.set_ylabel(r"$E_{paper}$ [V/mm]")
            ax.grid(alpha=0.2)
            if column == 0:
                ax.legend(fontsize=8, loc="best")
    fig.suptitle("Why the next iterations were rejected: raw force-balance fits", fontsize=15)
    fig.savefig(output / "failed_next_field_fits.png", dpi=220)
    fig.savefig(output / "failed_next_field_fits.pdf")
    plt.close(fig)


def plot_case_i_convergence(stages: dict, output: Path) -> None:
    selected = [
        stages[key] for key in sorted(stages)
        if key[0] == "case_I" and (key[1] == "constant" or key[1] == "linear")
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for metadata, data, path in selected:
        iteration = int(metadata["iteration"])
        label = f"iteration {iteration} ({metadata['branch']})"
        z = np.asarray(data["dust_z_paper_mm"])
        order = np.argsort(z)
        axes[0, 0].plot(z[order], -data["input_field_sim_V_m"][order] / 1e3, "o-", label=label)
        axes[0, 1].plot(
            np.arange(1, 8), 100.0 * data["normalized_force_imbalance"], "o-", label=label
        )
        axes[1, 0].errorbar(
            np.arange(1, 8),
            -data["mean_charge_C"] / ELEMENTARY_CHARGE_C / 1e4,
            yerr=data["std_charge_C"] / ELEMENTARY_CHARGE_C / 1e4,
            marker="o", label=label,
        )
    linear = [item for item in selected if item[0]["branch"] == "linear"]
    iterations = [item[0]["iteration"] for item in linear]
    changes = [100.0 * item[0]["relative_field_change"] for item in linear]
    rms = [100.0 * np.sqrt(np.mean(item[1]["normalized_force_imbalance"] ** 2)) for item in linear]
    axes[1, 1].semilogy(iterations, changes, "o-", label="max field change")
    axes[1, 1].semilogy(iterations, rms, "s-", label="RMS force imbalance")
    axes[1, 1].axhline(1.0, color="tab:red", ls="--", label="1% convergence target")
    axes[0, 0].set(xlabel=r"$z_{paper}$ [mm]", ylabel=r"imposed $E_{paper}$ [V/mm]")
    axes[0, 1].set(xlabel="dust index", ylabel=r"$F_T/(m_dg)$ [%]", xticks=np.arange(1, 8))
    axes[1, 0].set(xlabel="dust index", ylabel=r"$-Q/(10^4e)$", xticks=np.arange(1, 8))
    axes[1, 1].set(xlabel="linear iteration", ylabel="metric [%]", xticks=iterations)
    for ax in axes.flat:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("Case I linear branch: oscillatory convergence without relaxation", fontsize=15)
    fig.savefig(output / "case_I_linear_convergence.png", dpi=220)
    fig.savefig(output / "case_I_linear_convergence.pdf")
    plt.close(fig)


def plot_density_maps(stages: dict, output: Path) -> None:
    cases = ("case_I", "case_II", "case_III")
    fig, axes = plt.subplots(3, 3, figsize=(14, 11), constrained_layout=True)
    images = []
    for column, case in enumerate(cases):
        for row, branch in enumerate(("constant", "linear", "quadratic")):
            source = latest(stages, case, branch)
            ax = axes[row, column]
            if source is None:
                ax.axis("off")
                continue
            metadata, data, path = source
            fields_path = path.with_name(path.stem + "_fields.npz")
            with np.load(fields_path, allow_pickle=False) as fields:
                x = fields["x_m"] / lambda_de_m()
                z = fields["z_m"] / lambda_de_m()
                density = fields["ion_density_m3"] / REFERENCE_DENSITY_M3
            image = ax.pcolormesh(x, z, density, shading="auto", cmap="viridis", vmin=0.0, vmax=3.0)
            images.append(image)
            ax.plot(np.zeros(7), data["dust_z_sim_m"] / lambda_de_m(), "wo", ms=2.5)
            ax.set_title(f"{case.replace('_', ' ').title()} — {branch}, iter {metadata['iteration']}")
            ax.set(xlabel=r"$x/\lambda_{De}$", ylabel=r"$z_{sim}/\lambda_{De}$")
    if images:
        fig.colorbar(images[-1], ax=axes, label=r"$n_i/n_0$", shrink=0.85)
    fig.suptitle("Available time-averaged ion-density maps (clipped at 3 n0)", fontsize=15)
    fig.savefig(output / "available_density_maps.png", dpi=220)
    fig.savefig(output / "available_density_maps.pdf")
    plt.close(fig)


def plot_presheath_profiles(stages: dict, output: Path) -> None:
    cases = ("case_I", "case_II", "case_III")
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True, sharex="col")
    cs = math.sqrt(KB * ELECTRON_TEMPERATURE_K / ION_MASS_KG)
    for column, case in enumerate(cases):
        for branch, color in (("constant", "0.45"), ("linear", "tab:orange"), ("quadratic", "tab:purple")):
            source = latest(stages, case, branch)
            if source is None:
                continue
            metadata, data, _ = source
            z = data["background_z_m"] / lambda_de_m()
            density = np.mean(data["background_density_m3"], axis=1)
            flux = np.mean(data["background_axial_flux_m2_s"], axis=1)
            velocity = np.divide(flux, density, out=np.zeros_like(flux), where=density > 0.0)
            label = f"{branch} iter {metadata['iteration']}"
            axes[0, column].plot(z, density / REFERENCE_DENSITY_M3, color=color, label=label)
            axes[1, column].plot(z, velocity / cs, color=color, label=label)
        axes[0, column].set_title(case.replace("_", " ").title())
        axes[0, column].set_ylabel(r"$n_i/n_0$")
        axes[1, column].set_ylabel(r"$\langle v_z\rangle/C_s$")
        axes[1, column].set_xlabel(r"$z_{sim}/\lambda_{De}$")
        for ax in axes[:, column]:
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
    fig.suptitle("Dust-free presheath backgrounds for every available branch", fontsize=15)
    fig.savefig(output / "presheath_profiles.png", dpi=220)
    fig.savefig(output / "presheath_profiles.pdf")
    plt.close(fig)


def plot_constant_article_panels(stages: dict, output: Path) -> None:
    """Recreate article Figure 4(b,c) using only constant-field stages."""
    fig, axes = plt.subplots(2, 1, figsize=(8, 9.5), constrained_layout=True, sharex=True)
    dust_mass_kg = 5.56e-13
    gravity_m_s2 = 9.81
    for case in ("case_I", "case_II", "case_III"):
        metadata, data, _ = latest(stages, case, "constant")
        color, marker, label = CASE_STYLE[case]
        z = np.asarray(data["dust_z_paper_mm"])
        order = np.argsort(z)
        charge = -np.asarray(data["mean_charge_C"]) / ELEMENTARY_CHARGE_C / 1e4
        charge_error = np.asarray(data["std_charge_C"]) / ELEMENTARY_CHARGE_C / 1e4
        required = -np.asarray(data["required_field_sim_V_m"]) / 1e3

        burn = int(metadata["burn_in_step"])
        charge_history = np.asarray(data["charge_history_C"])[burn:]
        other_force = (
            np.asarray(data["orbit_force_history_N"])[burn:, :, 2]
            + np.asarray(data["collection_force_history_N"])[burn:, :, 2]
            + np.asarray(data["dust_dust_force_history_N"])[burn:, :, 2]
        )
        # The instantaneous orbit force is deliberately noisy.  Figure 4 uses
        # time-averaged forces, so show the scatter of 100 non-overlapping time
        # blocks instead of the much larger single-step fluctuation.
        block_count = min(100, len(charge_history))
        block_size = len(charge_history) // block_count
        used = block_count * block_size
        block_charge = charge_history[:used].reshape(block_count, block_size, 7).mean(axis=1)
        block_force = other_force[:used].reshape(block_count, block_size, 7).mean(axis=1)
        block_sim_field = -(
            dust_mass_kg * gravity_m_s2 + block_force
        ) / block_charge
        required_error = np.std(-block_sim_field / 1e3, axis=0, ddof=0)

        axes[0].errorbar(
            z[order], charge[order], yerr=charge_error[order], color=color,
            marker=marker, ms=9 if marker != "*" else 12, lw=1.8, capsize=4, label=label,
        )
        axes[1].errorbar(
            z[order], required[order], yerr=required_error[order], color=color,
            marker=marker, ms=8 if marker != "*" else 11, ls="none", capsize=4,
        )
        grid = np.linspace(4.5, 7.5, 300)
        linear = metadata["next_unweighted_fits"]["linear"]
        quadratic = metadata["next_unweighted_fits"]["quadratic"]
        axes[1].plot(grid, paper_polynomial(linear, grid), color=color, lw=2.0, ls="-", label=f"{label} linear")
        axes[1].plot(grid, paper_polynomial(quadratic, grid), color=color, lw=2.0, ls="--", label=f"{label} quadratic")

    axes[0].text(0.025, 0.93, "(b)", transform=axes[0].transAxes, fontsize=19, va="top")
    axes[1].text(0.025, 0.93, "(c)", transform=axes[1].transAxes, fontsize=19, va="top")
    axes[0].set_ylabel(r"Charge $(10^4\ e)$", fontsize=15)
    axes[1].set_ylabel(r"$E_z$ (V/mm)", fontsize=15)
    axes[1].set_xlabel(r"$z$ (mm)", fontsize=15)
    axes[0].set_xlim(4.0, 8.0)
    axes[0].set_ylim(1.0, 3.9)
    axes[1].set_ylim(-3.5, -0.5)
    axes[0].legend(frameon=False, fontsize=11, loc="upper left", bbox_to_anchor=(0.12, 0.98))
    axes[1].legend(frameon=False, fontsize=8, ncol=2, loc="lower right")
    for ax in axes:
        ax.tick_params(labelsize=12, direction="in", top=True, right=True)
        ax.grid(False)
    fig.suptitle("OpenDust constant-field iteration (article Figure 4b,c layout)", fontsize=15)
    fig.savefig(output / "constant_run_figure4_bc.png", dpi=240)
    fig.savefig(output / "constant_run_figure4_bc.pdf")
    plt.close(fig)


def write_summary(stages: dict, output: Path) -> None:
    summary = []
    for (case, branch, iteration), (metadata, data, path) in sorted(stages.items()):
        summary.append(
            {
                "case": case,
                "branch": branch,
                "iteration": iteration,
                "source": str(path),
                "relative_field_change_percent": 100.0 * metadata["relative_field_change"],
                "rms_force_imbalance_percent": 100.0
                * float(np.sqrt(np.mean(data["normalized_force_imbalance"] ** 2))),
                "max_absolute_force_imbalance_percent": 100.0
                * float(np.max(np.abs(data["normalized_force_imbalance"]))),
                "charge_1e4_e": (-data["mean_charge_C"] / ELEMENTARY_CHARGE_C / 1e4).tolist(),
                "paper_field": metadata["paper_field"],
                "next_unweighted_fits": metadata.get("next_unweighted_fits", {}),
            }
        )
    (output / "partial_results_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path, help="Downloaded results directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    stages = {}
    for path in stage_files(args.results):
        metadata, arrays = load_stage(path)
        key = (path.parents[1].name, path.parent.name, int(metadata["iteration"]))
        stages[key] = (metadata, arrays, path)
    if not stages:
        raise RuntimeError(f"no completed stage artifacts found under {args.results}")
    plot_failed_fits(stages, args.output)
    plot_case_i_convergence(stages, args.output)
    plot_density_maps(stages, args.output)
    plot_presheath_profiles(stages, args.output)
    plot_constant_article_panels(stages, args.output)
    write_summary(stages, args.output)
    print(f"wrote {len(stages)} completed stages to {args.output}")


if __name__ == "__main__":
    main()
