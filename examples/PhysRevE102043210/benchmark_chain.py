#!/usr/bin/env python3
"""Short, production-faithful benchmark of the Matthews dust-chain workload.

The default workload uses Case I, 65,536 ions, seven self-consistently charged
dust grains, the prepared cylindrical presheath background, production
collision/charging subcycling, and plane diagnostics.  It runs one warm-up and
one measured 32-step sample, which should take roughly one minute in total on
the GPU where the full production stage takes about 90 minutes.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
import math
from pathlib import Path
import statistics
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from opendust_jax import (
    ARTICLE_CASES,
    ARTICLE_DUST_Z_OVER_LAMBDA,
    ArgonNullCollisionModel,
    ChargingSubcyclingConfig,
    CollisionSubcyclingConfig,
    DustParticleSI,
    OpenDustJAX,
    PlaneDiagnosticConfig,
    SimulationConfigSI,
    load_presheath_background,
    presheath_preparation_speed_limit_m_s,
    recommend_main_timestep,
)
from opendust_jax.article_reproduction import (
    ARTICLE_DUST_MASS_KG,
    ARTICLE_DUST_RADIUS_M,
    ARTICLE_ELECTRON_TEMPERATURE_K,
    ARTICLE_REFERENCE_DENSITY_M3,
    PaperElectricField,
)
from opendust_jax.collisions import CollisionEvents

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from workflow import article_plasma, load_config  # noqa: E402


class _NoOpArgonNullCollisionModel(ArgonNullCollisionModel):
    """Keep the production collision grid but remove collision computation."""

    def apply(self, key, velocities_m_s, thermal_speed_m_s, dt_s):
        del key, thermal_speed_m_s, dt_s
        no_event = jnp.zeros((velocities_m_s.shape[0],), dtype=bool)
        return velocities_m_s, CollisionEvents(*(no_event for _ in range(6)))


class _BenchmarkOpenDustJAX(OpenDustJAX):
    """Production engine with an optional benchmark-only zero pair force."""

    def __init__(self, *args, benchmark_skip_pair_force: bool, **kwargs):
        self._benchmark_skip_pair_force = benchmark_skip_pair_force
        super().__init__(*args, **kwargs)

    def _pair_force(self, positions):
        if self._benchmark_skip_pair_force:
            return jnp.zeros_like(positions), jnp.asarray(0, dtype=jnp.int32)
        return super()._pair_force(positions)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=HERE / "article_config.json",
        help="Article workflow configuration (default: article_config.json).",
    )
    parser.add_argument(
        "--case-index", type=int, choices=range(len(ARTICLE_CASES)), default=0
    )
    parser.add_argument(
        "--background",
        type=Path,
        help=(
            "Prepared presheath NPZ or its prefix. By default, use the constant "
            "iteration-1 background below the configured output directory."
        ),
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=32,
        help="Measured outer steps; 32 is about 30 s for a 90 min full stage.",
    )
    parser.add_argument("--warmup", type=int, default=1, help="Untimed, identical runs.")
    parser.add_argument("--repeat", type=int, default=1, help="Measured repetitions.")
    parser.add_argument(
        "--label",
        default="default",
        help="Safe filename label used to retain several benchmark variants.",
    )
    parser.add_argument(
        "--component-mode",
        choices=("full", "no-pair", "no-collisions", "fast-only"),
        default="full",
        help=(
            "Select full physics or remove expensive components while retaining "
            "the same collision/charging time grid."
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100,
        help="Outer scan chunk size, matching production by default.",
    )
    parser.add_argument(
        "--charging-substeps",
        type=int,
        help="Override automatic charging substeps for controlled experiments.",
    )
    parser.add_argument(
        "--collision-substeps",
        type=int,
        help="Override automatic collision substeps; unsafe values are rejected.",
    )
    parser.add_argument(
        "--diagnostics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the production 81x161 plane diagnostic workload.",
    )
    parser.add_argument(
        "--diagnostic-interval",
        type=int,
        default=None,
        help="Override the production diagnostic interval.",
    )
    parser.add_argument(
        "--allow-non-cuda",
        action="store_true",
        help="Allow a non-GPU backend for functional checks only.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "results/chain_benchmark",
        help="Benchmark JSON and diagnostic output directory.",
    )
    return parser


def _resolved_background(config, case_name: str, requested: Path | None) -> Path:
    if requested is not None:
        source = requested.expanduser().resolve()
        if source.is_file():
            return source
        candidate = source.with_name(source.name + "_background.npz")
        return candidate if candidate.is_file() else source
    return (
        Path(config.output_dir)
        / case_name
        / "constant"
        / "iteration_01_presheath_background.npz"
    ).resolve()


def _sync_result(result) -> None:
    jax.block_until_ready(result.final_state.positions_m)


def main() -> None:
    args = _parser().parse_args()
    if args.steps <= 0 or args.warmup < 0 or args.repeat <= 0:
        raise SystemExit(
            "--steps and --repeat must be positive; --warmup cannot be negative"
        )
    allowed_label_characters = (
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
    )
    if not args.label or any(
        character not in allowed_label_characters for character in args.label
    ):
        raise SystemExit("--label may contain only letters, digits, '.', '_' and '-'")
    if args.chunk_size <= 0:
        raise SystemExit("--chunk-size must be positive")
    if args.diagnostic_interval is not None and args.diagnostic_interval <= 0:
        raise SystemExit("--diagnostic-interval must be positive")
    if jax.default_backend() != "gpu" and not args.allow_non_cuda:
        raise SystemExit(
            f"this benchmark requires the JAX GPU backend; got {jax.default_backend()} "
            "(use --allow-non-cuda only for a functional check)"
        )

    config = load_config(args.config.resolve())
    case = ARTICLE_CASES[args.case_index]
    lambda_de = math.sqrt(
        8.8541878128e-12
        * 1.380649e-23
        * ARTICLE_ELECTRON_TEMPERATURE_K
        / (1.602176634e-19**2 * ARTICLE_REFERENCE_DENSITY_M3)
    )
    radius_m = config.radius_screening_lengths * lambda_de
    height_m = config.height_screening_lengths * lambda_de
    dust_z_m = np.asarray(ARTICLE_DUST_Z_OVER_LAMBDA) * lambda_de
    field = PaperElectricField(-case.constant_field_V_m / 1.0e3)
    plasma = article_plasma(
        field,
        case.initial_mach,
        height_m,
        float(dust_z_m[0]),
        float(dust_z_m[-1]),
    )

    speed_limit = presheath_preparation_speed_limit_m_s(plasma)
    collision_model_class = (
        _NoOpArgonNullCollisionModel
        if args.component_mode in ("no-collisions", "fast-only")
        else ArgonNullCollisionModel
    )
    collision_model = collision_model_class(
        plasma, maximum_relative_speed_m_s=speed_limit
    )
    collision_subcycling = CollisionSubcyclingConfig(
        substeps=args.collision_substeps
    )
    recommendation = recommend_main_timestep(
        plasma,
        collision_model=collision_model,
        collision_subcycling=collision_subcycling,
    )
    dt_s = config.production_timestep_omega_pi / recommendation.ion_plasma_frequency_s
    charging_subcycling = ChargingSubcyclingConfig(substeps=args.charging_substeps)

    background_path = _resolved_background(config, case.name, args.background)
    if not background_path.is_file():
        raise SystemExit(
            f"prepared background not found: {background_path}\n"
            "Pass --background PATH or first complete the matching constant "
            "background preparation."
        )
    background = load_presheath_background(background_path)

    initial_charge_C = -ARTICLE_DUST_MASS_KG * 9.81 / case.constant_field_V_m
    dust = [
        DustParticleSI(
            position_m=(0.0, 0.0, float(z)),
            radius_m=ARTICLE_DUST_RADIUS_M,
            initial_charge_C=initial_charge_C,
            charge_mode="self_consistent",
        )
        for z in dust_z_m
    ]
    force_backend = (
        "jax"
        if jax.default_backend() != "gpu" and args.allow_non_cuda
        else config.force_backend
    )
    simulation = SimulationConfigSI(
        radius_m=radius_m,
        height_m=height_m,
        particle_count=config.particle_count,
        steps=args.steps,
        dt_s=dt_s,
        collision_subcycling=collision_subcycling,
        charging_subcycling=charging_subcycling,
        seed=case.seed,
        force_backend=force_backend,
        step_chunk_size=args.chunk_size,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_interval = args.diagnostic_interval or config.diagnostic_interval
    diagnostics = (
        PlaneDiagnosticConfig(
            nx=config.diagnostic_nx,
            nz=config.diagnostic_nz,
            slice_thickness_m=0.1 * lambda_de,
            start_step=1,
            interval=diagnostic_interval,
            output_prefix=(
                args.output_dir / f"chain_benchmark_{args.label}_diagnostic"
            ),
        )
        if args.diagnostics
        else None
    )
    setup_started = time.perf_counter()
    skip_pair_force = args.component_mode in ("no-pair", "fast-only")
    collision_computation_enabled = args.component_mode not in (
        "no-collisions",
        "fast-only",
    )
    engine = _BenchmarkOpenDustJAX(
        plasma,
        simulation,
        dust,
        background=background,
        collision_model=collision_model,
        diagnostics=diagnostics,
        benchmark_skip_pair_force=skip_pair_force,
    )
    engine_setup_s = time.perf_counter() - setup_started

    print(f"JAX devices: {jax.devices()}", flush=True)
    print(
        "Matthews chain benchmark: "
        f"case={case.name}, ions={config.particle_count}, dust={len(dust)}, "
        f"steps={args.steps}, dt={dt_s:.6e} s, "
        f"Ncoll={engine.collision_substeps}, Ncharge={engine.charging_substeps}, "
        f"Nsegments={engine.integration_segments}, mode={args.component_mode}, "
        f"diagnostics={args.diagnostics}",
        flush=True,
    )
    print(f"engine setup: {engine_setup_s:.3f} s", flush=True)

    warmup_times: list[float] = []
    for index in range(args.warmup):
        started = time.perf_counter()
        result = engine.run()
        _sync_result(result)
        elapsed = time.perf_counter() - started
        warmup_times.append(elapsed)
        print(f"warmup {index + 1}/{args.warmup}: {elapsed:.3f} s", flush=True)

    measured_times: list[float] = []
    final_result = None
    for index in range(args.repeat):
        started = time.perf_counter()
        final_result = engine.run()
        _sync_result(final_result)
        elapsed = time.perf_counter() - started
        measured_times.append(elapsed)
        print(f"measured {index + 1}/{args.repeat}: {elapsed:.3f} s", flush=True)

    median_s = statistics.median(measured_times)
    full_steps = math.ceil(
        config.duration_plasma_periods
        * 2.0
        * math.pi
        / config.production_timestep_omega_pi
    )
    diagnostic_samples = (
        1 + (args.steps - 1) // diagnostic_interval if args.diagnostics else 0
    )
    report = {
        "format_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "devices": [str(device) for device in jax.devices()],
        "case": asdict(case),
        "background": str(background_path),
        "background_fingerprint": background.fingerprint,
        "component_mode": args.component_mode,
        "pair_force_enabled": not skip_pair_force,
        "collision_computation_enabled": collision_computation_enabled,
        "simulation": asdict(simulation),
        "resolved": {
            "ion_plasma_frequency_s": recommendation.ion_plasma_frequency_s,
            "dt_s": dt_s,
            "collision_substeps": engine.collision_substeps,
            "charging_substeps": engine.charging_substeps,
            "integration_segments": engine.integration_segments,
            "maximum_fast_dt_s": engine.maximum_fast_dt_s,
            # initial_state() evaluates the complete force once, and run()
            # initializes its scan carry with another pair/fast evaluation.
            "scheduled_pair_force_calls_including_initialization": args.steps + 2,
            "executed_pair_force_calls": (
                0 if skip_pair_force else args.steps + 2
            ),
            "fast_force_calls_including_initialization": (
                args.steps * engine.integration_segments + 2
            ),
            "scheduled_collision_operator_calls": (
                args.steps * engine.collision_substeps
            ),
            "executed_collision_computations": (
                0
                if not collision_computation_enabled
                else args.steps * engine.collision_substeps
            ),
            "diagnostic_samples": diagnostic_samples,
            "full_production_steps": full_steps,
        },
        "timings": {
            "engine_setup_s": engine_setup_s,
            "warmup_s": warmup_times,
            "measured_s": measured_times,
            "median_s": median_s,
            "seconds_per_outer_step": median_s / args.steps,
            "projected_full_production_s": median_s * full_steps / args.steps,
        },
        "collision_statistics": asdict(final_result.collision_statistics),
    }
    report_path = args.output_dir / f"chain_benchmark_{args.label}.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    print(
        f"median: {median_s:.3f} s; {median_s / args.steps:.6f} s/outer-step; "
        "linear full-stage projection: "
        f"{report['timings']['projected_full_production_s'] / 60:.1f} min",
        flush=True,
    )
    print(f"report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
