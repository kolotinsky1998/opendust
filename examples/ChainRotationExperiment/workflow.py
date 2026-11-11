#!/usr/bin/env python3
"""Iterative OpenDust calculation for the rotated experimental chain."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import jax
import numpy as np


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from opendust_jax import (  # noqa: E402
    ArgonNullCollisionModel,
    ArgonPresheathPlasmaSI,
    ChargingSubcyclingConfig,
    CollisionSubcyclingConfig,
    DustParticleSI,
    OpenDustJAX,
    OutputConfig,
    PlaneDiagnosticConfig,
    PresheathEquilibrationConfig,
    QuadraticElectricFieldSI,
    SimulationConfigSI,
    prepare_presheath_background,
    presheath_preparation_speed_limit_m_s,
    recommend_main_timestep,
)
from opendust_jax.article_reproduction import (  # noqa: E402
    article_drift_mach,
    artifact_fingerprint,
    completed_artifact_matches,
    fit_required_field_constrained,
    save_iteration_artifact,
)
from opendust_jax.correlation_analysis import (  # noqa: E402
    estimate_correlated_mean,
)
from opendust_jax.models import (  # noqa: E402
    BOLTZMANN_J_K,
    ELEMENTARY_CHARGE_C,
    EPSILON_0_F_M,
)


@dataclass(frozen=True)
class ChainConfig:
    output_dir: Path
    coordinate_csv: Path
    electron_temperatures_eV: tuple[float, ...]
    reference_densities_m3: tuple[float, ...]
    pressure_Pa: float
    neutral_temperature_K: float
    ion_mass_kg: float
    dust_diameter_m: float
    dust_density_kg_m3: float
    gravity_m_s2: float
    initial_mach: float
    iterations: int
    production_timestep_omega_pi: float
    duration_plasma_periods: float
    burn_in_plasma_periods: float
    radial_buffer_screening_lengths: float
    minimum_height_screening_lengths: float
    upstream_buffer_screening_lengths: float
    downstream_buffer_screening_lengths: float
    low_density_particle_count: int
    default_particle_count: int
    low_density_threshold_m3: float
    force_backend: str
    diagnostic_planes: tuple[str, ...]
    diagnostic_nx: int
    diagnostic_nz: int
    diagnostic_interval: int
    diagnostic_slice_screening_lengths: float
    step_chunk_size: int
    background_grid_spacing_screening_lengths: float
    background_density_tolerance: float
    background_flux_tolerance: float
    background_consecutive_blocks: int
    background_minimum_transit_times: float
    background_block_transit_times: float
    background_maximum_transit_times: float
    background_averaging_blocks: int
    background_minimum_samples_per_convergence_window: int
    base_seed: int

    @property
    def dust_radius_m(self) -> float:
        return 0.5 * self.dust_diameter_m

    @property
    def dust_mass_kg(self) -> float:
        return (
            4.0
            / 3.0
            * math.pi
            * self.dust_radius_m**3
            * self.dust_density_kg_m3
        )


@dataclass(frozen=True)
class PlasmaCase:
    index: int
    electron_temperature_eV: float
    electron_temperature_K: float
    reference_density_m3: float
    name: str
    seed: int


@dataclass(frozen=True)
class ChainGeometry:
    particle_ids: np.ndarray
    laboratory_positions_m: np.ndarray
    simulation_positions_m: np.ndarray
    screening_length_m: float
    radius_m: float
    height_m: float
    chain_span_m: float
    maximum_radial_offset_m: float


def load_config(path: Path) -> ChainConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    base = path.resolve().parent
    for name in ("output_dir", "coordinate_csv"):
        value = Path(raw[name])
        raw[name] = value if value.is_absolute() else (base / value).resolve()
    raw["electron_temperatures_eV"] = tuple(raw["electron_temperatures_eV"])
    raw["reference_densities_m3"] = tuple(raw["reference_densities_m3"])
    raw["diagnostic_planes"] = tuple(raw["diagnostic_planes"])
    config = ChainConfig(**raw)
    if config.iterations != 5:
        raise ValueError("the chain workflow requires exactly five stages")
    if config.force_backend not in {"cuda", "jax"}:
        raise ValueError("force_backend must be 'cuda' or 'jax'")
    return config


def cases(config: ChainConfig) -> tuple[PlasmaCase, ...]:
    result = []
    for temperature_eV in config.electron_temperatures_eV:
        temperature_K = temperature_eV * ELEMENTARY_CHARGE_C / BOLTZMANN_J_K
        for density in config.reference_densities_m3:
            index = len(result)
            density_power = int(round(math.log10(density)))
            result.append(
                PlasmaCase(
                    index=index,
                    electron_temperature_eV=temperature_eV,
                    electron_temperature_K=temperature_K,
                    reference_density_m3=density,
                    name=f"Te_{temperature_eV:g}eV_n_1e{density_power}",
                    seed=config.base_seed + index,
                )
            )
    return tuple(result)


def read_particle_coordinates(path: Path) -> tuple[np.ndarray, np.ndarray]:
    required = ("particle_id", "x_m", "y_m", "z_m")
    rows: list[tuple[int, float, float, float]] = []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = set(required) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"coordinate CSV is missing columns: {sorted(missing)}")
        for row in reader:
            rows.append(
                (
                    int(row["particle_id"]),
                    float(row["x_m"]),
                    float(row["y_m"]),
                    float(row["z_m"]),
                )
            )
    if not rows:
        raise ValueError("coordinate CSV contains no particles")
    identifiers = np.asarray([row[0] for row in rows], dtype=np.int64)
    positions = np.asarray([row[1:] for row in rows], dtype=np.float64)
    if len(set(identifiers.tolist())) != len(identifiers):
        raise ValueError("particle_id values must be unique")
    if not np.all(np.isfinite(positions)):
        raise ValueError("particle coordinates must be finite")
    order = np.argsort(positions[:, 2], kind="stable")
    return identifiers[order], positions[order]


def screening_length_m(case: PlasmaCase) -> float:
    return math.sqrt(
        EPSILON_0_F_M
        * BOLTZMANN_J_K
        * case.electron_temperature_K
        / (ELEMENTARY_CHARGE_C**2 * case.reference_density_m3)
    )


def build_geometry(config: ChainConfig, case: PlasmaCase) -> ChainGeometry:
    particle_ids, laboratory = read_particle_coordinates(config.coordinate_csv)
    lam = screening_length_m(case)
    centered = laboratory.copy()
    centered[:, :2] -= np.mean(centered[:, :2], axis=0)
    radial = np.linalg.norm(centered[:, :2], axis=1)
    maximum_radial = float(np.max(radial))
    span = float(np.ptp(centered[:, 2]))
    radius = maximum_radial + config.radial_buffer_screening_lengths * lam
    requested_height = span + (
        config.upstream_buffer_screening_lengths
        + config.downstream_buffer_screening_lengths
    ) * lam
    height = max(config.minimum_height_screening_lengths * lam, requested_height)
    top_z = -0.5 * height + config.upstream_buffer_screening_lengths * lam
    centered[:, 2] = top_z + (laboratory[:, 2] - np.min(laboratory[:, 2]))
    if np.max(centered[:, 2]) > 0.5 * height - config.downstream_buffer_screening_lengths * lam + 1e-15:
        raise ValueError("resolved domain does not preserve the downstream buffer")
    return ChainGeometry(
        particle_ids,
        laboratory,
        centered,
        lam,
        radius,
        height,
        span,
        maximum_radial,
    )


def particle_count(config: ChainConfig, case: PlasmaCase) -> int:
    if case.reference_density_m3 <= config.low_density_threshold_m3:
        return config.low_density_particle_count
    return config.default_particle_count


def constant_field_for_mach(config: ChainConfig, case: PlasmaCase) -> float:
    """Invert the article mobility relation monotonically by bisection."""

    def residual(field: float) -> float:
        return article_drift_mach(
            field,
            pressure_Pa=config.pressure_Pa,
            neutral_temperature_K=config.neutral_temperature_K,
            electron_temperature_K=case.electron_temperature_K,
            ion_mass_kg=config.ion_mass_kg,
        ) - config.initial_mach

    lower, upper = 0.0, 1.0
    while residual(upper) < 0.0:
        upper *= 2.0
        if upper > 1.0e8:
            raise RuntimeError("could not bracket the constant field for the requested Mach")
    for _ in range(80):
        midpoint = 0.5 * (lower + upper)
        if residual(midpoint) < 0.0:
            lower = midpoint
        else:
            upper = midpoint
    return 0.5 * (lower + upper)


def case_directory(config: ChainConfig, case: PlasmaCase) -> Path:
    return config.output_dir / case.name


def artifact_prefix(config: ChainConfig, case: PlasmaCase, iteration: int) -> Path:
    suffix = "constant" if iteration == 1 else "quadratic"
    return case_directory(config, case) / f"iteration_{iteration:02d}_{suffix}"


def read_stage(prefix: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    metadata = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
    with np.load(prefix.with_suffix(".npz"), allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]) for name in data.files}
    return metadata, arrays


def coordinate_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stage_field_and_state(
    config: ChainConfig,
    case: PlasmaCase,
    geometry: ChainGeometry,
    iteration: int,
) -> tuple[QuadraticElectricFieldSI, float, np.ndarray, str | None, dict | None]:
    z = geometry.simulation_positions_m[:, 2]
    if iteration == 1:
        value = constant_field_for_mach(config, case)
        field = QuadraticElectricFieldSI(
            E0_V_m=value,
            z_min_m=float(z[0]),
            z_max_m=float(z[-1]),
        )
        charge = -config.dust_mass_kg * config.gravity_m_s2 / value
        return field, config.initial_mach, np.full(len(z), charge), None, None

    previous_prefix = artifact_prefix(config, case, iteration - 1)
    previous_metadata, previous = read_stage(previous_prefix)
    fit = fit_required_field_constrained(
        z,
        previous["required_field_sim_V_m"],
        geometry.height_m,
        2,
        active_z_min_m=float(z[0]),
        active_z_max_m=float(z[-1]),
    )
    field = fit.field.to_simulation(
        geometry.height_m,
        z_min_m=float(z[0]),
        z_max_m=float(z[-1]),
    )
    field.validate_for_domain(geometry.height_m, lowest_dust_z_m=float(z[0]))
    mach = article_drift_mach(
        abs(field.value_V_m(float(z[0]))),
        pressure_Pa=config.pressure_Pa,
        neutral_temperature_K=config.neutral_temperature_K,
        electron_temperature_K=case.electron_temperature_K,
        ion_mass_kg=config.ion_mass_kg,
    )
    return (
        field,
        mach,
        previous["mean_charge_C"],
        previous_metadata["fingerprint"],
        asdict(fit),
    )


def time_grid(config: ChainConfig, omega_pi_s: float) -> tuple[float, int, int, float]:
    plasma_period = 2.0 * math.pi / omega_pi_s
    dt_s = config.production_timestep_omega_pi / omega_pi_s
    steps = math.ceil(config.duration_plasma_periods * plasma_period / dt_s)
    burn_in_step = math.ceil(config.burn_in_plasma_periods * plasma_period / dt_s)
    return dt_s, steps, burn_in_step, plasma_period


def correlated_summary(values: np.ndarray, dt_s: float) -> dict[str, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    output_shape = values.shape[1:]
    result = {
        "mean": np.empty(output_shape),
        "std": np.empty(output_shape),
        "sem": np.empty(output_shape),
        "confidence_low": np.empty(output_shape),
        "confidence_high": np.empty(output_shape),
        "tau_int_s": np.empty(output_shape),
        "effective_sample_size": np.empty(output_shape),
        "stationarity_warning": np.empty(output_shape, dtype=bool),
    }
    for index in np.ndindex(output_shape):
        estimate = estimate_correlated_mean(values[(slice(None),) + index], dt_s)
        result["mean"][index] = estimate.mean
        result["std"][index] = estimate.standard_deviation
        result["sem"][index] = estimate.standard_error
        result["confidence_low"][index] = estimate.confidence_low
        result["confidence_high"][index] = estimate.confidence_high
        result["tau_int_s"][index] = estimate.integrated_autocorrelation_time_s
        result["effective_sample_size"][index] = estimate.effective_sample_size
        result["stationarity_warning"][index] = estimate.stationarity_warning
    return result


def summarize_result(
    result,
    burn_in_step: int,
    dt_s: float,
    input_field_V_m: np.ndarray,
    config: ChainConfig,
) -> dict[str, np.ndarray]:
    histories = {
        "charge": np.asarray(result.dust_charges_C)[burn_in_step:],
        "orbit_force": np.asarray(result.dust_orbit_forces_N)[burn_in_step:],
        "collection_force": np.asarray(result.dust_collection_forces_N)[burn_in_step:],
        "dust_dust_force": np.asarray(result.dust_dust_forces_N)[burn_in_step:],
        "external_force": np.asarray(result.dust_external_forces_N)[burn_in_step:],
    }
    gravity = np.zeros_like(histories["orbit_force"])
    gravity[:, :, 2] = config.dust_mass_kg * config.gravity_m_s2
    histories["gravity_force"] = gravity
    histories["total_force"] = (
        histories["orbit_force"]
        + histories["collection_force"]
        + histories["dust_dust_force"]
        + histories["external_force"]
        + gravity
    )
    summary: dict[str, np.ndarray] = {}
    for name, values in histories.items():
        for statistic, statistic_values in correlated_summary(values, dt_s).items():
            summary[f"{statistic}_{name}{'_C' if name == 'charge' else '_N'}"] = statistic_values
    nonexternal_z = (
        summary["mean_orbit_force_N"][:, 2]
        + summary["mean_collection_force_N"][:, 2]
        + summary["mean_dust_dust_force_N"][:, 2]
    )
    mean_charge = summary["mean_charge_C"]
    summary["required_field_sim_V_m"] = -(
        config.dust_mass_kg * config.gravity_m_s2 + nonexternal_z
    ) / mean_charge
    weight = config.dust_mass_kg * config.gravity_m_s2
    summary["normalized_force_imbalance"] = (
        weight + nonexternal_z + mean_charge * input_field_V_m
    ) / weight
    summary["relative_charge_sem"] = np.divide(
        np.abs(summary["sem_charge_C"]),
        np.abs(mean_charge),
        out=np.full_like(mean_charge, np.inf),
        where=mean_charge != 0.0,
    )
    return summary


def run_stage(
    config: ChainConfig, case: PlasmaCase, geometry: ChainGeometry, iteration: int
) -> Path:
    field, mach, initial_charges, parent, applied_fit = stage_field_and_state(
        config, case, geometry, iteration
    )
    z = geometry.simulation_positions_m[:, 2]
    plasma = ArgonPresheathPlasmaSI(
        neutral_temperature_K=config.neutral_temperature_K,
        electron_temperature_K=case.electron_temperature_K,
        reference_density_m3=case.reference_density_m3,
        pressure_Pa=config.pressure_Pa,
        inlet_mach=mach,
        ion_mass_kg=config.ion_mass_kg,
        electric_field=field,
    )
    speed_limit = presheath_preparation_speed_limit_m_s(plasma)
    collision_model = ArgonNullCollisionModel(
        plasma, maximum_relative_speed_m_s=speed_limit
    )
    collision_subcycling = CollisionSubcyclingConfig()
    recommendation = recommend_main_timestep(
        plasma,
        collision_model=collision_model,
        collision_subcycling=collision_subcycling,
    )
    dt_s, steps, burn_in_step, plasma_period_s = time_grid(
        config, recommendation.ion_plasma_frequency_s
    )
    count = particle_count(config, case)
    volume = math.pi * geometry.radius_m**2 * geometry.height_m
    real_ions = case.reference_density_m3 * volume
    superion_weight = real_ions / count
    input_field = np.asarray([field.value_V_m(float(value)) for value in z])
    payload = {
        "config": {**asdict(config), "output_dir": str(config.output_dir), "coordinate_csv": str(config.coordinate_csv)},
        "coordinate_sha256": coordinate_sha256(config.coordinate_csv),
        "case": asdict(case),
        "geometry": {
            "screening_length_m": geometry.screening_length_m,
            "radius_m": geometry.radius_m,
            "height_m": geometry.height_m,
            "simulation_positions_m": geometry.simulation_positions_m.tolist(),
        },
        "iteration": iteration,
        "field": asdict(field),
        "mach": mach,
        "initial_charges_C": initial_charges.tolist(),
        "parent": parent,
        "dt_s": dt_s,
        "steps": steps,
        "burn_in_step": burn_in_step,
    }
    fingerprint = artifact_fingerprint(payload)
    prefix = artifact_prefix(config, case, iteration)
    if completed_artifact_matches(prefix, fingerprint):
        print(f"skip compatible completed stage: {prefix}", flush=True)
        return prefix

    preparation_dt = recommend_main_timestep(
        plasma, collision_model=collision_model
    ).recommended_dt_s
    common = dict(
        radius_m=geometry.radius_m,
        height_m=geometry.height_m,
        particle_count=count,
        seed=case.seed,
        force_backend=config.force_backend,
        step_chunk_size=config.step_chunk_size,
    )
    preparation = SimulationConfigSI(
        steps=0,
        dt_s=preparation_dt,
        collision_subcycling=None,
        **common,
    )
    background_prefix = prefix.with_name(prefix.name + "_presheath")
    print(f"preparing background for {case.name}, iteration {iteration}", flush=True)
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
            block_transit_times=config.background_block_transit_times,
            maximum_transit_times=config.background_maximum_transit_times,
            averaging_blocks=config.background_averaging_blocks,
            source_block_size=min(4096, count),
            minimum_samples_per_convergence_window=config.background_minimum_samples_per_convergence_window,
        ),
        collision_model=collision_model,
    )
    jax.block_until_ready(background.potential_V)
    dust = [
        DustParticleSI(
            position_m=tuple(map(float, position)),
            radius_m=config.dust_radius_m,
            initial_charge_C=float(charge),
            charge_mode="self_consistent",
        )
        for position, charge in zip(geometry.simulation_positions_m, initial_charges)
    ]
    production = SimulationConfigSI(
        steps=steps,
        dt_s=dt_s,
        collision_subcycling=collision_subcycling,
        charging_subcycling=ChargingSubcyclingConfig(),
        **common,
    )
    diagnostics = PlaneDiagnosticConfig(
        nx=config.diagnostic_nx,
        nz=config.diagnostic_nz,
        slice_thickness_m=config.diagnostic_slice_screening_lengths
        * geometry.screening_length_m,
        start_step=burn_in_step,
        interval=config.diagnostic_interval,
        output_prefix=prefix,
        planes=config.diagnostic_planes,
    )
    print(f"running {case.name}, iteration {iteration}/5, Mach={mach:.6g}", flush=True)
    engine = OpenDustJAX(
        plasma,
        production,
        dust,
        background=background,
        collision_model=collision_model,
        diagnostics=diagnostics,
        output=OutputConfig(
            checkpoint_path=prefix.with_name(prefix.name + "_restart.npz")
        ),
    )
    result = engine.run()
    jax.block_until_ready(result.final_state.positions_m)
    summary = summarize_result(result, burn_in_step, dt_s, input_field, config)
    next_fit = fit_required_field_constrained(
        z,
        summary["required_field_sim_V_m"],
        geometry.height_m,
        2,
        active_z_min_m=float(z[0]),
        active_z_max_m=float(z[-1]),
    )
    next_simulation_field = next_fit.field.to_simulation(
        geometry.height_m,
        z_min_m=float(z[0]),
        z_max_m=float(z[-1]),
    )
    next_values = np.asarray(
        [next_simulation_field.value_V_m(float(value)) for value in z]
    )
    relative_field_change = float(
        np.max(
            np.abs(next_values - input_field)
            / np.maximum(np.abs(input_field), np.finfo(float).tiny)
        )
    )
    arrays = {
        **summary,
        "particle_ids": geometry.particle_ids,
        "laboratory_positions_m": geometry.laboratory_positions_m,
        "simulation_positions_m": geometry.simulation_positions_m,
        "input_field_sim_V_m": input_field,
        "times_s": np.asarray(result.times_s),
        "charge_history_C": np.asarray(result.dust_charges_C),
        "orbit_force_history_N": np.asarray(result.dust_orbit_forces_N),
        "collection_force_history_N": np.asarray(result.dust_collection_forces_N),
        "dust_dust_force_history_N": np.asarray(result.dust_dust_forces_N),
        "external_force_history_N": np.asarray(result.dust_external_forces_N),
        "gravity_force_history_N": np.broadcast_to(
            np.asarray([0.0, 0.0, config.dust_mass_kg * config.gravity_m_s2]),
            np.asarray(result.dust_external_forces_N).shape,
        ),
        "total_force_history_N": (
            np.asarray(result.dust_orbit_forces_N)
            + np.asarray(result.dust_collection_forces_N)
            + np.asarray(result.dust_dust_forces_N)
            + np.asarray(result.dust_external_forces_N)
            + np.broadcast_to(
                np.asarray([0.0, 0.0, config.dust_mass_kg * config.gravity_m_s2]),
                np.asarray(result.dust_external_forces_N).shape,
            )
        ),
        "background_r_m": background.r_m,
        "background_z_m": background.z_m,
        "background_density_m3": background.density_m3,
        "background_axial_flux_m2_s": background.axial_flux_m2_s,
        "background_axial_velocity_m_s": background.axial_velocity_m_s,
    }
    metadata = {
        "status": "complete",
        "fingerprint": fingerprint,
        **payload,
        "applied_fit": applied_fit,
        "next_quadratic_fit": asdict(next_fit),
        "next_simulation_field": asdict(next_simulation_field),
        "relative_field_change": relative_field_change,
        "converged_at_one_percent": relative_field_change <= 0.01,
        "screening_length_m": geometry.screening_length_m,
        "radius_m": geometry.radius_m,
        "height_m": geometry.height_m,
        "chain_span_m": geometry.chain_span_m,
        "maximum_radial_offset_m": geometry.maximum_radial_offset_m,
        "dust_mass_kg": config.dust_mass_kg,
        "particle_count": count,
        "real_ion_count": real_ions,
        "superion_weight": superion_weight,
        "superion_charge_C": ELEMENTARY_CHARGE_C * superion_weight,
        "superion_to_initial_dust_charge_ratio": float(
            ELEMENTARY_CHARGE_C * superion_weight / np.mean(np.abs(initial_charges))
        ),
        "ion_plasma_frequency_s": recommendation.ion_plasma_frequency_s,
        "ion_plasma_period_s": plasma_period_s,
        "burn_in_step": burn_in_step,
        "resolved_collision_substeps": engine.collision_substeps,
        "resolved_charging_substeps": engine.charging_substeps,
        "diagnostic_planes": list(config.diagnostic_planes),
        "charge_sem_above_one_percent": bool(
            np.any(summary["relative_charge_sem"] > 0.01)
        ),
        "stationarity_warning": bool(
            any(
                np.any(value)
                for name, value in summary.items()
                if name.startswith("stationarity_warning_")
            )
        ),
        "force_collection_model": (
            "OpenDust physical absorption radius and orbit+collection force; "
            "DRIAD b_c/Maxwell-stress construction is not used"
        ),
    }
    save_iteration_artifact(prefix, arrays, metadata)
    print(
        f"completed {case.name} iteration {iteration}: "
        f"max dE/E={relative_field_change:.3%}, "
        f"max charge SEM={np.max(summary['relative_charge_sem']):.3%}",
        flush=True,
    )
    return prefix


def run_case(config: ChainConfig, case_index: int) -> None:
    selected = cases(config)
    if not 0 <= case_index < len(selected):
        raise ValueError(f"case index must lie in [0, {len(selected) - 1}]")
    case = selected[case_index]
    geometry = build_geometry(config, case)
    for iteration in range(1, config.iterations + 1):
        run_stage(config, case, geometry, iteration)


def smoke_config(config: ChainConfig) -> ChainConfig:
    return replace(
        config,
        output_dir=config.output_dir / "smoke",
        iterations=5,
        duration_plasma_periods=0.4,
        burn_in_plasma_periods=0.1,
        low_density_particle_count=64,
        default_particle_count=64,
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
        background_block_transit_times=0.05,
        background_maximum_transit_times=5.0,
        background_averaging_blocks=1,
        background_minimum_samples_per_convergence_window=1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "chain_config.json")
    parser.add_argument("--case-index", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--describe-grid", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.output_dir is not None:
        config = replace(config, output_dir=args.output_dir.resolve())
    if args.describe_grid:
        for case in cases(config):
            geometry = build_geometry(config, case)
            print(
                case.index,
                case.name,
                f"lambda_De={geometry.screening_length_m:.6e} m",
                f"R={geometry.radius_m:.6e} m",
                f"H={geometry.height_m:.6e} m",
                f"N={particle_count(config, case)}",
            )
        return
    if args.case_index is None:
        parser.error("--case-index is required unless --describe-grid is used")
    if args.smoke:
        config = smoke_config(config)
    run_case(config, args.case_index)


if __name__ == "__main__":
    main()
