#!/usr/bin/env python3
"""CUDA/Slurm workflow for figures 3--7 of Phys. Rev. E 102, 043210."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import sys

import numpy as np

from opendust_jax import (
    ARTICLE_CASES,
    ARTICLE_DUST_Z_OVER_LAMBDA,
    ArgonPresheathPlasmaSI,
    ArticleReproductionConfig,
    ChargingSubcyclingConfig,
    CollisionSubcyclingConfig,
    DustParticleSI,
    OpenDustJAX,
    OutputConfig,
    PaperElectricField,
    PlaneDiagnosticConfig,
    PresheathEquilibrationConfig,
    SimulationConfigSI,
    prepare_presheath_background,
    recommend_main_timestep,
)
from opendust_jax.article_reproduction import (
    ARTICLE_DUST_MASS_KG,
    ARTICLE_DUST_RADIUS_M,
    ARTICLE_ELECTRON_TEMPERATURE_K,
    ARTICLE_ION_MASS_KG,
    ARTICLE_ION_TEMPERATURE_K,
    ARTICLE_PRESSURE_PA,
    ARTICLE_REFERENCE_DENSITY_M3,
    article_drift_mach,
    artifact_fingerprint,
    completed_artifact_matches,
    config_as_metadata,
    fit_required_field,
    save_iteration_artifact,
    summarize_production_result,
)


HERE = Path(__file__).resolve().parent


def load_config(path: Path) -> ArticleReproductionConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    output = Path(raw.pop("output_dir"))
    if not output.is_absolute():
        output = (path.parent / output).resolve()
    return ArticleReproductionConfig(output_dir=output, **raw)


def article_plasma(
    field: PaperElectricField,
    mach: float,
    height_m: float,
    z_min_m: float,
    z_max_m: float,
):
    return ArgonPresheathPlasmaSI(
        neutral_temperature_K=ARTICLE_ION_TEMPERATURE_K,
        electron_temperature_K=ARTICLE_ELECTRON_TEMPERATURE_K,
        reference_density_m3=ARTICLE_REFERENCE_DENSITY_M3,
        pressure_Pa=ARTICLE_PRESSURE_PA,
        inlet_mach=mach,
        ion_mass_kg=ARTICLE_ION_MASS_KG,
        electric_field=field.to_simulation(
            height_m,
            z_min_m=z_min_m,
            z_max_m=z_max_m,
            enforce_presheath_constraints=True,
        ),
    )


def case_directory(config: ArticleReproductionConfig, case_index: int) -> Path:
    return Path(config.output_dir) / ARTICLE_CASES[case_index].name


def artifact_prefix(
    config: ArticleReproductionConfig,
    case_index: int,
    branch: str,
    iteration: int,
) -> Path:
    return case_directory(config, case_index) / branch / f"iteration_{iteration:02d}"


def read_stage(prefix: Path) -> tuple[dict, dict[str, np.ndarray]]:
    metadata = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
    with np.load(prefix.with_suffix(".npz"), allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]) for name in data.files}
    return metadata, arrays


def stage_field_and_mach(
    config: ArticleReproductionConfig,
    case_index: int,
    branch: str,
    iteration: int,
    height_m: float,
    dust_z_m: np.ndarray,
) -> tuple[PaperElectricField, float, np.ndarray, str | None]:
    case = ARTICLE_CASES[case_index]
    if iteration == 1:
        return (
            PaperElectricField(-case.constant_field_V_m / 1.0e3),
            case.initial_mach,
            np.full(len(dust_z_m), -ARTICLE_DUST_MASS_KG * 9.81 / case.constant_field_V_m),
            None,
        )
    previous_prefix = (
        artifact_prefix(config, case_index, "constant", 1)
        if iteration == 2
        else artifact_prefix(config, case_index, branch, iteration - 1)
    )
    previous_metadata, previous = read_stage(previous_prefix)
    degree = 1 if branch == "linear" else 2
    field = fit_required_field(
        dust_z_m,
        previous["required_field_sim_V_m"],
        height_m,
        degree,
    )
    z_upstream_paper_mm = (0.5 * height_m - dust_z_m[0]) * 1.0e3
    mach = article_drift_mach(abs(field.value_V_mm(z_upstream_paper_mm)) * 1.0e3)
    return field, mach, previous["mean_charge_C"], previous_metadata["fingerprint"]


def run_stage(
    config: ArticleReproductionConfig,
    case_index: int,
    branch: Literal["constant", "linear", "quadratic"],
    iteration: int,
) -> Path:
    if branch == "constant" and iteration != 1:
        raise ValueError("the constant branch has only iteration 1")
    if branch != "constant" and not 2 <= iteration <= config.iterations:
        raise ValueError("iterative branches start at 2 and end at configured iterations")
    case = ARTICLE_CASES[case_index]
    lambda_de = math.sqrt(
        8.8541878128e-12
        * 1.380649e-23
        * ARTICLE_ELECTRON_TEMPERATURE_K
        / (1.602176634e-19**2 * ARTICLE_REFERENCE_DENSITY_M3)
    )
    radius_m = config.radius_screening_lengths * lambda_de
    height_m = config.height_screening_lengths * lambda_de
    dust_z_m = np.asarray(ARTICLE_DUST_Z_OVER_LAMBDA) * lambda_de
    field, mach, initial_charges, parent = stage_field_and_mach(
        config, case_index, branch, iteration, height_m, dust_z_m
    )
    if config.smoke_project_invalid_fields and branch != "constant":
        try:
            candidate = field.to_simulation(
                height_m,
                z_min_m=float(dust_z_m[0]),
                z_max_m=float(dust_z_m[-1]),
            )
            candidate.validate_for_domain(height_m)
        except ValueError:
            field = (
                PaperElectricField(-4.0, 0.2, 0.0)
                if branch == "linear"
                else PaperElectricField(0.5, -1.32, 0.12)
            )
            parent = f"{parent}: smoke field projected into strict admissible set"
    plasma = article_plasma(
        field, mach, height_m, float(dust_z_m[0]), float(dust_z_m[-1])
    )
    collision_subcycling = CollisionSubcyclingConfig()
    recommendation = recommend_main_timestep(
        plasma, collision_subcycling=collision_subcycling
    )
    omega_pi = recommendation.ion_plasma_frequency_s
    dt_s = recommendation.recommended_dt_s
    steps = math.ceil(
        config.duration_plasma_periods * 2.0 * math.pi / (omega_pi * dt_s)
    )
    burn_in_step = math.ceil(
        config.burn_in_plasma_periods * 2.0 * math.pi / (omega_pi * dt_s)
    )
    input_payload = {
        "workflow": config_as_metadata(config),
        "case": asdict(case),
        "branch": branch,
        "iteration": iteration,
        "paper_field": asdict(field),
        "mach": mach,
        "parent": parent,
        "dt_s": dt_s,
        "steps": steps,
        "collision_subcycling": asdict(collision_subcycling),
    }
    fingerprint = artifact_fingerprint(input_payload)
    prefix = artifact_prefix(config, case_index, branch, iteration)
    if completed_artifact_matches(prefix, fingerprint):
        print(f"skip compatible completed stage: {prefix}")
        return prefix

    common = dict(
        radius_m=radius_m,
        height_m=height_m,
        particle_count=config.particle_count,
        dt_s=dt_s,
        collision_subcycling=collision_subcycling,
        seed=case.seed,
        force_backend=config.force_backend,
        step_chunk_size=config.step_chunk_size,
    )
    preparation = SimulationConfigSI(steps=0, **common)
    background_prefix = prefix.parent / f"{prefix.name}_presheath"
    background = prepare_presheath_background(
        plasma,
        preparation,
        PresheathEquilibrationConfig(
            output_prefix=background_prefix,
            grid_spacing_screening_lengths=config.background_grid_spacing_screening_lengths,
            density_tolerance=config.background_density_tolerance,
            flux_tolerance=config.background_flux_tolerance,
            consecutive_blocks=config.background_consecutive_blocks,
            minimum_transit_times=config.background_minimum_transit_times,
            maximum_transit_times=config.background_maximum_transit_times,
            averaging_blocks=config.background_averaging_blocks,
            source_block_size=min(4096, config.particle_count),
        ),
    )
    dust = [
        DustParticleSI(
            position_m=(0.0, 0.0, float(z)),
            radius_m=ARTICLE_DUST_RADIUS_M,
            initial_charge_C=float(charge),
            charge_mode="self_consistent",
        )
        for z, charge in zip(dust_z_m, initial_charges)
    ]
    production = SimulationConfigSI(
        steps=steps,
        charging_subcycling=ChargingSubcyclingConfig(),
        **common,
    )
    diagnostics = PlaneDiagnosticConfig(
        nx=config.diagnostic_nx,
        nz=config.diagnostic_nz,
        slice_thickness_m=0.1 * lambda_de,
        start_step=burn_in_step,
        interval=config.diagnostic_interval,
        output_prefix=prefix,
    )
    engine = OpenDustJAX(
        plasma,
        production,
        dust,
        background=background,
        diagnostics=diagnostics,
        output=OutputConfig(checkpoint_path=prefix.with_name(prefix.name + "_restart.npz")),
    )
    result = engine.run()
    input_field_sim = np.asarray(
        [plasma.electric_field.value_V_m(float(z)) for z in dust_z_m]
    )
    summary = summarize_production_result(result, burn_in_step, input_field_sim)
    next_fits = {}
    fit_degrees = (1, 2) if branch == "constant" else ((1,) if branch == "linear" else (2,))
    relative_field_change = float(
        np.max(
            np.abs(summary["required_field_sim_V_m"] - input_field_sim)
            / np.maximum(np.abs(input_field_sim), np.finfo(float).tiny)
        )
    )
    for degree in fit_degrees:
        fitted = fit_required_field(
            dust_z_m, summary["required_field_sim_V_m"], height_m, degree
        )
        fit_name = "linear" if degree == 1 else "quadratic"
        next_fits[fit_name] = asdict(fitted)
        if fit_name == branch:
            next_simulation = fitted.to_simulation(
                height_m,
                z_min_m=float(dust_z_m[0]),
                z_max_m=0.5 * height_m,
                enforce_presheath_constraints=False,
            )
            next_values = np.asarray(
                [next_simulation.value_V_m(float(z)) for z in dust_z_m]
            )
            relative_field_change = float(
                np.max(
                    np.abs(next_values - input_field_sim)
                    / np.maximum(np.abs(input_field_sim), np.finfo(float).tiny)
                )
            )
    arrays = {
        **summary,
        "dust_z_sim_m": dust_z_m,
        "dust_z_paper_mm": (0.5 * height_m - dust_z_m) * 1.0e3,
        "input_field_sim_V_m": input_field_sim,
        "times_s": np.asarray(result.times_s),
        "charge_history_C": np.asarray(result.dust_charges_C),
        "orbit_force_history_N": np.asarray(result.dust_orbit_forces_N),
        "collection_force_history_N": np.asarray(result.dust_collection_forces_N),
        "dust_dust_force_history_N": np.asarray(result.dust_dust_forces_N),
        "background_r_m": background.r_m,
        "background_z_m": background.z_m,
        "background_density_m3": background.density_m3,
        "background_axial_flux_m2_s": background.axial_flux_m2_s,
        "background_axial_velocity_m_s": background.axial_velocity_m_s,
    }
    status = "complete"
    converged = bool(relative_field_change <= 0.01) if branch != "constant" else False
    metadata = {
        "status": status,
        "fingerprint": fingerprint,
        **input_payload,
        "background_fingerprint": background.fingerprint,
        "background_prefix": str(background_prefix),
        "force_collection_model": (
            "OpenDust physical absorption radius and orbit+collection force; "
            "DRIAD b_c/Maxwell-stress construction is intentionally not used"
        ),
        "burn_in_step": burn_in_step,
        "resolved_collision_substeps": engine.collision_substeps,
        "dt_collision_s": engine.collision_dt_s,
        "resolved_charging_substeps": engine.charging_substeps,
        "dt_charge_s": engine.charging_dt_s,
        "integration_segments": engine.integration_segments,
        "maximum_fast_dt_s": engine.maximum_fast_dt_s,
        "pair_force_outer_updates": steps,
        "pair_force_calls_including_initialization": steps + 1,
        "collision_statistics": asdict(result.collision_statistics),
        "relative_field_change": float(relative_field_change),
        "converged_at_one_percent": converged,
        "next_unweighted_fits": next_fits,
        "units": {
            "charge": "C",
            "force": "N",
            "field": "V/m",
            "paper_field_coefficients": "V/mm, V/mm^2, V/mm^3",
        },
    }
    save_iteration_artifact(prefix, arrays, metadata)
    print(
        f"completed {case.name} {branch} iteration {iteration}: "
        f"dt_ii={dt_s:.4e}s, Ncoll={engine.collision_substeps}, "
        f"Ncharge={engine.charging_substeps}, Nsegments={engine.integration_segments}, "
        f"pair calls={steps}+1 init, "
        f"max dE/E={relative_field_change:.3%}"
    )
    return prefix


def run_branch(config, case_index: int, branch: Literal["linear", "quadratic"]):
    constant = artifact_prefix(config, case_index, "constant", 1)
    if not constant.with_suffix(".json").is_file():
        raise FileNotFoundError(f"constant stage must finish first: {constant}")
    for iteration in range(2, config.iterations + 1):
        run_stage(config, case_index, branch, iteration)


def smoke_config(config: ArticleReproductionConfig) -> ArticleReproductionConfig:
    return replace(
        config,
        output_dir=Path(config.output_dir) / "smoke",
        particle_count=64,
        iterations=2,
        duration_plasma_periods=0.4,
        burn_in_plasma_periods=0.1,
        force_backend="jax",
        diagnostic_nx=9,
        diagnostic_nz=13,
        diagnostic_interval=1,
        step_chunk_size=1,
        background_grid_spacing_screening_lengths=0.5,
        background_density_tolerance=1.0,
        background_flux_tolerance=1.0,
        background_consecutive_blocks=1,
        background_minimum_transit_times=0.5,
        background_maximum_transit_times=5.0,
        background_averaging_blocks=1,
        smoke_project_invalid_fields=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=HERE / "article_config.json")
    parser.add_argument(
        "--stage", choices=("constant", "linear", "quadratic", "smoke"), required=True
    )
    parser.add_argument("--case-index", type=int, choices=range(3))
    args = parser.parse_args()
    config = load_config(args.config)
    if args.stage == "smoke":
        config = smoke_config(config)
        for index in range(3):
            run_stage(config, index, "constant", 1)
            run_branch(config, index, "linear")
            run_branch(config, index, "quadratic")
        return
    if args.case_index is None:
        parser.error("--case-index is required outside smoke mode")
    if args.stage == "constant":
        run_stage(config, args.case_index, "constant", 1)
    else:
        run_branch(config, args.case_index, args.stage)


if __name__ == "__main__":
    main()
