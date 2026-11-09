#!/usr/bin/env python3
"""CUDA/Slurm workflow for figures 3--7 of Phys. Rev. E 102, 043210."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import time
from typing import Literal

import jax
import numpy as np

from opendust_jax import (
    ARTICLE_CASES,
    ARTICLE_DUST_Z_OVER_LAMBDA,
    ArgonPresheathPlasmaSI,
    ArgonNullCollisionModel,
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
    presheath_preparation_speed_limit_m_s,
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
    fit_required_field_constrained,
    resolve_article_time_grid,
    save_iteration_artifact,
    summarize_production_result,
)


HERE = Path(__file__).resolve().parent


def format_wall_time(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


@contextmanager
def timed_phase(label: str, metadata_key: str, timings: dict[str, float]):
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    started = time.perf_counter()
    print(f"[{started_at}] starting {label}", flush=True)
    try:
        yield
    except BaseException:
        elapsed = time.perf_counter() - started
        failed_at = datetime.now().astimezone().isoformat(timespec="seconds")
        print(
            f"[{failed_at}] failed {label} after {elapsed:.3f} s "
            f"({format_wall_time(elapsed)})",
            flush=True,
        )
        raise
    elapsed = time.perf_counter() - started
    timings[metadata_key] = elapsed
    finished_at = datetime.now().astimezone().isoformat(timespec="seconds")
    print(
        f"[{finished_at}] completed {label} in {elapsed:.3f} s "
        f"({format_wall_time(elapsed)})",
        flush=True,
    )


def file_signature(path: Path) -> tuple[int, int] | None:
    if not path.is_file():
        return None
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


def update_saved_metadata(path: Path, metadata: dict) -> None:
    temporary = path.with_name(path.name + ".timing.tmp")
    temporary.write_text(
        json.dumps({"format_version": 1, **metadata}, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


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
) -> tuple[PaperElectricField, float, np.ndarray, str | None, dict | None]:
    case = ARTICLE_CASES[case_index]
    if iteration == 1:
        return (
            PaperElectricField(-case.constant_field_V_m / 1.0e3),
            case.initial_mach,
            np.full(len(dust_z_m), -ARTICLE_DUST_MASS_KG * 9.81 / case.constant_field_V_m),
            None,
            None,
        )
    previous_prefix = (
        artifact_prefix(config, case_index, "constant", 1)
        if iteration == 2
        else artifact_prefix(config, case_index, branch, iteration - 1)
    )
    previous_metadata, previous = read_stage(previous_prefix)
    degree = 1 if branch == "linear" else 2
    fit = fit_required_field_constrained(
        dust_z_m,
        previous["required_field_sim_V_m"],
        height_m,
        degree,
        active_z_min_m=float(dust_z_m[0]),
        active_z_max_m=float(dust_z_m[-1]),
    )
    field = fit.field
    z_upstream_paper_mm = (0.5 * height_m - dust_z_m[0]) * 1.0e3
    mach = article_drift_mach(abs(field.value_V_mm(z_upstream_paper_mm)) * 1.0e3)
    return (
        field,
        mach,
        previous["mean_charge_C"],
        previous_metadata["fingerprint"],
        asdict(fit),
    )


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
    field, mach, initial_charges, parent, applied_fit = stage_field_and_mach(
        config, case_index, branch, iteration, height_m, dust_z_m
    )
    plasma = article_plasma(
        field, mach, height_m, float(dust_z_m[0]), float(dust_z_m[-1])
    )
    collision_subcycling = CollisionSubcyclingConfig()
    recommendation = recommend_main_timestep(
        plasma, collision_subcycling=collision_subcycling
    )
    omega_pi = recommendation.ion_plasma_frequency_s
    time_grid = resolve_article_time_grid(config, omega_pi)
    dt_s = time_grid.dt_s
    steps = time_grid.outer_steps
    burn_in_step = time_grid.burn_in_step
    preparation_speed_limit = presheath_preparation_speed_limit_m_s(plasma)
    preparation_collision = ArgonNullCollisionModel(
        plasma, maximum_relative_speed_m_s=preparation_speed_limit
    )
    preparation_timestep = recommend_main_timestep(
        plasma, collision_model=preparation_collision
    )
    input_payload = {
        "workflow": config_as_metadata(config),
        "case": asdict(case),
        "branch": branch,
        "iteration": iteration,
        "paper_field": asdict(field),
        "applied_constrained_fit": applied_fit,
        "mach": mach,
        "parent": parent,
        "dt_s": dt_s,
        "steps": steps,
        "dt_ii_s": dt_s,
        "outer_steps": steps,
        "ion_plasma_frequency_s": omega_pi,
        "ion_plasma_period_s": time_grid.ion_plasma_period_s,
        "duration_s": time_grid.duration_s,
        "plasma_limit_s": recommendation.plasma_limit_s,
        "collision_subcycling": asdict(collision_subcycling),
        "background_preparation": {
            "dt_s": preparation_timestep.recommended_dt_s,
            "maximum_relative_speed_m_s": preparation_speed_limit,
            "maximum_energy_eV": preparation_collision.maximum_majorant_energy_eV,
            "maximum_frequency_s": preparation_collision.maximum_frequency_s,
        },
    }
    fingerprint = artifact_fingerprint(input_payload)
    prefix = artifact_prefix(config, case_index, branch, iteration)
    if completed_artifact_matches(prefix, fingerprint):
        print(f"skip compatible completed stage: {prefix}")
        return prefix

    stage_started = time.perf_counter()
    timings: dict[str, float] = {}

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
    preparation = SimulationConfigSI(
        radius_m=radius_m,
        height_m=height_m,
        particle_count=config.particle_count,
        steps=0,
        dt_s=preparation_timestep.recommended_dt_s,
        collision_subcycling=None,
        seed=case.seed,
        force_backend=config.force_backend,
        step_chunk_size=config.step_chunk_size,
    )
    background_prefix = prefix.parent / f"{prefix.name}_presheath"
    background_npz = background_prefix.with_name(
        background_prefix.name + "_background.npz"
    )
    background_signature_before = file_signature(background_npz)
    with timed_phase(
        f"{case.name} {branch} iteration {iteration} background preparation",
        "background_preparation_wall_s",
        timings,
    ):
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
            collision_model=preparation_collision,
        )
        jax.block_until_ready(background.potential_V)
    background_cache_reused = (
        background_signature_before is not None
        and background_signature_before == file_signature(background_npz)
    )
    print(
        "presheath background: "
        + ("compatible cache reused" if background_cache_reused else "calculated and saved"),
        flush=True,
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
    with timed_phase(
        f"{case.name} {branch} iteration {iteration} production",
        "production_wall_s",
        timings,
    ):
        engine = OpenDustJAX(
            plasma,
            production,
            dust,
            background=background,
            diagnostics=diagnostics,
            output=OutputConfig(
                checkpoint_path=prefix.with_name(prefix.name + "_restart.npz")
            ),
        )
        result = engine.run()
        jax.block_until_ready(result.final_state.positions_m)
    postprocessing_started = time.perf_counter()
    postprocessing_started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"[{postprocessing_started_at}] starting postprocessing", flush=True)
    input_field_sim = np.asarray(
        [plasma.electric_field.value_V_m(float(z)) for z in dust_z_m]
    )
    summary = summarize_production_result(result, burn_in_step, input_field_sim)
    next_fits = {}
    next_constrained_fits = {}
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
        constrained = fit_required_field_constrained(
            dust_z_m,
            summary["required_field_sim_V_m"],
            height_m,
            degree,
            active_z_min_m=float(dust_z_m[0]),
            active_z_max_m=float(dust_z_m[-1]),
        )
        next_constrained_fits[fit_name] = asdict(constrained)
        if fit_name == branch:
            next_simulation = constrained.field.to_simulation(
                height_m,
                z_min_m=float(dust_z_m[0]),
                z_max_m=float(dust_z_m[-1]),
            )
            next_simulation.validate_for_domain(
                height_m, lowest_dust_z_m=float(dust_z_m[0])
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
        "background_cache_reused": background_cache_reused,
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
        "next_constrained_fits": next_constrained_fits,
        "units": {
            "charge": "C",
            "force": "N",
            "field": "V/m",
            "paper_field_coefficients": "V/mm, V/mm^2, V/mm^3",
        },
    }
    metadata["timings"] = dict(timings)
    _, metadata_path = save_iteration_artifact(prefix, arrays, metadata)
    timings["postprocessing_wall_s"] = time.perf_counter() - postprocessing_started
    timings["total_stage_wall_s"] = time.perf_counter() - stage_started
    metadata["timings"] = dict(timings)
    update_saved_metadata(metadata_path, metadata)
    postprocessing_finished_at = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )
    print(
        f"[{postprocessing_finished_at}] completed postprocessing in "
        f"{timings['postprocessing_wall_s']:.3f} s "
        f"({format_wall_time(timings['postprocessing_wall_s'])})",
        flush=True,
    )
    print(
        f"completed {case.name} {branch} iteration {iteration}: "
        f"dt_ii={dt_s:.4e}s, Ncoll={engine.collision_substeps}, "
        f"Ncharge={engine.charging_substeps}, Nsegments={engine.integration_segments}, "
        f"pair calls={steps}+1 init, "
        f"max dE/E={relative_field_change:.3%}, "
        f"wall={format_wall_time(timings['total_stage_wall_s'])}"
    )
    return prefix


def run_branch(config, case_index: int, branch: Literal["linear", "quadratic"]):
    constant = artifact_prefix(config, case_index, "constant", 1)
    if not constant.with_suffix(".json").is_file():
        raise FileNotFoundError(f"constant stage must finish first: {constant}")
    branch_started = time.perf_counter()
    stored_stage_seconds = 0.0
    for iteration in range(2, config.iterations + 1):
        prefix = run_stage(config, case_index, branch, iteration)
        stage_metadata = json.loads(
            prefix.with_suffix(".json").read_text(encoding="utf-8")
        )
        stored_stage_seconds += float(
            stage_metadata.get("timings", {}).get("total_stage_wall_s", 0.0)
        )
    branch_wall = time.perf_counter() - branch_started
    print(
        f"completed {ARTICLE_CASES[case_index].name} {branch} branch: "
        f"current invocation={branch_wall:.3f} s ({format_wall_time(branch_wall)}), "
        f"sum of stored iteration timings={stored_stage_seconds:.3f} s "
        f"({format_wall_time(stored_stage_seconds)})",
        flush=True,
    )


def smoke_config(config: ArticleReproductionConfig) -> ArticleReproductionConfig:
    return replace(
        config,
        output_dir=Path(config.output_dir) / "smoke",
        particle_count=64,
        iterations=2,
        outer_steps=32,
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
        smoke_project_invalid_fields=False,
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
