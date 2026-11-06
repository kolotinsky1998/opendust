#!/usr/bin/env python3
"""Run the three dust-free Case-II profiles used in article Figure 3.

The fields and inlet Mach numbers are taken directly from Table I.  The
published Figure 3 curves are available only graphically, so the bundled
reference CSV is an approximate digitization and is always labelled as such.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from opendust_jax import (
    ARTICLE_DUST_Z_OVER_LAMBDA,
    ArgonPresheathPlasmaSI,
    PaperElectricField,
    PresheathEquilibrationConfig,
    SimulationConfigSI,
    prepare_presheath_background,
    recommend_main_timestep,
)
from opendust_jax.article_reproduction import (
    ARTICLE_ELECTRON_TEMPERATURE_K,
    ARTICLE_ION_MASS_KG,
    ARTICLE_ION_TEMPERATURE_K,
    ARTICLE_PRESSURE_PA,
    ARTICLE_REFERENCE_DENSITY_M3,
)


HERE = Path(__file__).resolve().parent
REFERENCE_CSV = HERE / "figure3_reference_digitized.csv"
BOLTZMANN_J_K = 1.380649e-23
EPSILON_0_F_M = 8.8541878128e-12
ELEMENTARY_CHARGE_C = 1.602176634e-19
PROFILE_ORDER = ("constant", "linear", "quadratic")
TABLE_I_CASE_II = {
    "constant": {"field": PaperElectricField(-16.00), "mach": 0.920},
    "linear": {"field": PaperElectricField(-4.092, 0.206), "mach": 0.322},
    "quadratic": {"field": PaperElectricField(-0.798, -1.001, 0.108), "mach": 0.300},
}


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _atomic_npz(path: Path, **arrays: Any) -> None:
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _article_geometry(config_path: Path) -> tuple[dict, float, float, float, np.ndarray]:
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    screening_length = math.sqrt(
        EPSILON_0_F_M
        * BOLTZMANN_J_K
        * ARTICLE_ELECTRON_TEMPERATURE_K
        / (ELEMENTARY_CHARGE_C**2 * ARTICLE_REFERENCE_DENSITY_M3)
    )
    radius = float(raw["radius_screening_lengths"]) * screening_length
    height = float(raw["height_screening_lengths"]) * screening_length
    dust_z = np.asarray(ARTICLE_DUST_Z_OVER_LAMBDA, dtype=np.float64) * screening_length
    return raw, screening_length, radius, height, dust_z


def _area_average(values: np.ndarray, r_m: np.ndarray) -> np.ndarray:
    """Cross-section average of a node-centred axisymmetric field."""
    integrand = values * r_m[None, :]
    numerator = np.sum(
        0.5 * (integrand[:, 1:] + integrand[:, :-1]) * np.diff(r_m)[None, :],
        axis=1,
    )
    denominator = 0.5 * r_m[-1] ** 2
    return numerator / denominator


def _profile_paths(output: Path, name: str) -> tuple[Path, Path, Path]:
    prefix = output / name
    return (
        prefix.with_name(prefix.name + "_background.npz"),
        prefix.with_name(prefix.name + "_profile.npz"),
        prefix.with_name(prefix.name + "_profile.json"),
    )


def run_profile(
    name: str,
    config_path: Path,
    output: Path,
    *,
    smoke: bool,
) -> None:
    raw, screening_length, radius, height, dust_z = _article_geometry(config_path)
    specification = TABLE_I_CASE_II[name]
    paper_field: PaperElectricField = specification["field"]
    # Paper z grows upstream whereas OpenDust z grows with the flow.  Above the
    # upper grain the field is clamped; downstream it remains polynomial to the
    # lower simulation boundary.  Rounded Case-II quadratic coefficients put
    # the vertex just outside the dust interval, so strict profile validation
    # is intentionally disabled only for these published reference fields.
    strict_validation = name != "quadratic"
    field = paper_field.to_simulation(
        height,
        z_min_m=float(dust_z[0]),
        z_max_m=0.5 * height,
        enforce_presheath_constraints=strict_validation,
    )
    plasma = ArgonPresheathPlasmaSI(
        neutral_temperature_K=ARTICLE_ION_TEMPERATURE_K,
        electron_temperature_K=ARTICLE_ELECTRON_TEMPERATURE_K,
        reference_density_m3=ARTICLE_REFERENCE_DENSITY_M3,
        pressure_Pa=ARTICLE_PRESSURE_PA,
        inlet_mach=float(specification["mach"]),
        ion_mass_kg=ARTICLE_ION_MASS_KG,
        electric_field=field,
    )
    timestep = recommend_main_timestep(plasma)
    particle_count = 64 if smoke else int(raw["particle_count"])
    simulation = SimulationConfigSI(
        radius_m=radius,
        height_m=height,
        particle_count=particle_count,
        steps=0,
        dt_s=timestep.recommended_dt_s,
        seed=20_041,
        force_backend="jax" if smoke else str(raw["force_backend"]),
        step_chunk_size=1 if smoke else int(raw["step_chunk_size"]),
    )
    if smoke:
        equilibration = PresheathEquilibrationConfig(
            output_prefix=output / name,
            grid_spacing_screening_lengths=0.5,
            density_tolerance=10.0,
            flux_tolerance=10.0,
            consecutive_blocks=1,
            minimum_transit_times=0.05,
            block_transit_times=0.05,
            maximum_transit_times=0.5,
            averaging_blocks=1,
            source_block_size=64,
        )
    else:
        equilibration = PresheathEquilibrationConfig(
            output_prefix=output / name,
            grid_spacing_screening_lengths=float(raw["background_grid_spacing_screening_lengths"]),
            density_tolerance=float(raw["background_density_tolerance"]),
            flux_tolerance=float(raw["background_flux_tolerance"]),
            consecutive_blocks=int(raw["background_consecutive_blocks"]),
            minimum_transit_times=float(raw["background_minimum_transit_times"]),
            maximum_transit_times=float(raw["background_maximum_transit_times"]),
            averaging_blocks=int(raw["background_averaging_blocks"]),
            source_block_size=min(4096, particle_count),
        )
    background = prepare_presheath_background(plasma, simulation, equilibration)
    density = _area_average(background.density_m3, background.r_m)
    flux = _area_average(background.axial_flux_m2_s, background.r_m)
    velocity = np.divide(flux, density, out=np.zeros_like(flux), where=density > 0.0)
    sound_speed = math.sqrt(
        BOLTZMANN_J_K * ARTICLE_ELECTRON_TEMPERATURE_K / ARTICLE_ION_MASS_KG
    )
    z_paper_mm = (0.5 * height - background.z_m) * 1.0e3
    order = np.argsort(z_paper_mm)
    _, profile_npz, profile_json = _profile_paths(output, name)
    _atomic_npz(
        profile_npz,
        z_paper_mm=z_paper_mm[order],
        density_over_n0=(density / ARTICLE_REFERENCE_DENSITY_M3)[order],
        velocity_over_cs=(velocity / sound_speed)[order],
        density_m3=density[order],
        axial_flux_m2_s=flux[order],
        velocity_m_s=velocity[order],
    )
    _atomic_json(
        profile_json,
        {
            "status": "complete",
            "profile": name,
            "table_I_case_II": {
                "paper_field": asdict(paper_field),
                "mach": specification["mach"],
            },
            "field_policy": "constant above upper dust; polynomial downstream to lower boundary",
            "strict_validation": strict_validation,
            "strict_validation_reason": (
                None
                if strict_validation
                else (
                    "Table-I coefficients are rounded and the Case-II quadratic vertex lies just "
                    "outside the dust interval"
                )
            ),
            "mode": "dust-free, noninteracting-ion presheath equilibration with null collisions",
            "particle_count": particle_count,
            "dt_s": timestep.recommended_dt_s,
            "ion_plasma_frequency_s": timestep.ion_plasma_frequency_s,
            "collision_majorant_frequency_s": timestep.collision_frequency_s,
            "background_fingerprint": background.fingerprint,
            "background_metadata": background.metadata,
            "smoke": smoke,
        },
    )
    print(
        f"completed Figure 3 {name}: N={particle_count}, dt={timestep.recommended_dt_s:.6e}s, "
        f"background={background.fingerprint}",
        flush=True,
    )


def _load_reference() -> dict[str, dict[str, np.ndarray]]:
    rows: dict[str, list[tuple[float, float, float]]] = {name: [] for name in PROFILE_ORDER}
    with REFERENCE_CSV.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            rows[row["profile"]].append(
                (float(row["z_paper_mm"]), float(row["density_over_n0"]), float(row["velocity_over_cs"]))
            )
    return {
        name: {
            "z": np.asarray(values)[:, 0],
            "density": np.asarray(values)[:, 1],
            "velocity": np.asarray(values)[:, 2],
        }
        for name, values in rows.items()
    }


def plot_comparison(output: Path, dust_z_paper_mm: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reference = _load_reference()
    styles = {"constant": "-", "linear": "--", "quadratic": ":"}
    colors = {"constant": "#3155b7", "linear": "#d55e00", "quadratic": "#7b3294"}
    computed: dict[str, dict[str, np.ndarray]] = {}
    metrics: dict[str, dict[str, float]] = {}
    comparison_rows = []
    for name in PROFILE_ORDER:
        _, profile_npz, _ = _profile_paths(output, name)
        if not profile_npz.is_file():
            raise FileNotFoundError(f"missing {profile_npz}; run profile {name} first")
        with np.load(profile_npz, allow_pickle=False) as archive:
            computed[name] = {
                "z": np.asarray(archive["z_paper_mm"]),
                "density": np.asarray(archive["density_over_n0"]),
                "velocity": np.asarray(archive["velocity_over_cs"]),
            }
        ref = reference[name]
        calc_density = np.interp(ref["z"], computed[name]["z"], computed[name]["density"])
        calc_velocity = np.interp(ref["z"], computed[name]["z"], computed[name]["velocity"])
        metrics[name] = {
            "density_rmse": float(np.sqrt(np.mean((calc_density - ref["density"]) ** 2))),
            "velocity_rmse": float(np.sqrt(np.mean((calc_velocity - ref["velocity"]) ** 2))),
        }
        comparison_rows.extend(
            zip(
                [name] * len(ref["z"]), ref["z"], ref["density"], calc_density,
                ref["velocity"], calc_velocity,
            )
        )

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 8.5), sharex=True, constrained_layout=True)
    for name in PROFILE_ORDER:
        calc, ref = computed[name], reference[name]
        label = name.capitalize()
        axes[0].plot(calc["z"], calc["density"], styles[name], color=colors[name], lw=2.4, label=f"OpenDust {label}")
        axes[1].plot(calc["z"], calc["velocity"], styles[name], color=colors[name], lw=2.4, label=f"OpenDust {label}")
        axes[0].plot(ref["z"], ref["density"], "x", color=colors[name], ms=5, alpha=0.75, label=f"Article {label} (digitized)")
        axes[1].plot(ref["z"], ref["velocity"], "x", color=colors[name], ms=5, alpha=0.75, label=f"Article {label} (digitized)")
    lower, upper = float(np.min(dust_z_paper_mm)), float(np.max(dust_z_paper_mm))
    for axis in axes:
        axis.axvspan(lower, upper, color="#d9ddf3", alpha=0.55, zorder=-5)
        axis.axvline(lower, color="0.35", lw=0.8)
        axis.axvline(upper, color="0.35", lw=0.8)
        axis.set_xlim(4.0, 8.0)
        axis.grid(alpha=0.16)
    axes[0].set_ylabel(r"$n_i/n_{i0}$")
    axes[1].set_ylabel(r"$v_d/C_s$")
    axes[1].set_xlabel(r"$z_{paper}$ [mm]")
    axes[0].text(0.025, 0.93, "(a)", transform=axes[0].transAxes, fontsize=16)
    axes[1].text(0.025, 0.08, "(b)", transform=axes[1].transAxes, fontsize=16)
    axes[0].legend(fontsize=7.5, ncol=2, loc="best")
    axes[1].legend(fontsize=7.5, ncol=2, loc="best")
    fig.suptitle("Figure 3, Case II: OpenDust vs digitized article curves")
    fig.savefig(output / "figure3_comparison.png", dpi=240)
    fig.savefig(output / "figure3_comparison.pdf")
    plt.close(fig)

    with (output / "figure3_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "profile", "z_paper_mm", "article_density_over_n0", "opendust_density_over_n0",
                "article_velocity_over_cs", "opendust_velocity_over_cs",
            ]
        )
        writer.writerows(comparison_rows)
    _atomic_json(
        output / "figure3_validation.json",
        {
            "status": "complete",
            "metrics": metrics,
            "reference": (
                "Approximate pixel digitization of published Figure 3 supplied with this example; "
                "it is not original author data and has an estimated reading uncertainty of order 1-2%."
            ),
        },
    )
    print(f"wrote Figure 3 comparison to {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=HERE / "article_config.json")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results" / "figure3_table_I")
    parser.add_argument("--profile", choices=(*PROFILE_ORDER, "all"), default="all")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.smoke:
        args.output_dir = args.output_dir / "smoke"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw, _, _, height, dust_z = _article_geometry(args.config)
    del raw
    selected = PROFILE_ORDER if args.profile == "all" else (args.profile,)
    if not args.plot_only:
        for name in selected:
            run_profile(name, args.config, args.output_dir, smoke=args.smoke)
    if args.profile == "all" or args.plot_only:
        dust_z_paper_mm = (0.5 * height - dust_z) * 1.0e3
        plot_comparison(args.output_dir, dust_z_paper_mm)


if __name__ == "__main__":
    main()
