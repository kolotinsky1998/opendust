#!/usr/bin/env python3
"""Run the Piel2017 Figure 1 case with averaged x-z plane diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import jax
import numpy as np

# This standalone simulation always uses FP64 temporal accumulators for the
# averaged diagnostics.  Configure JAX before importing opendust_jax modules or
# creating any arrays, so callers do not need a separate environment variable.
jax.config.update("jax_enable_x64", True)


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from opendust_jax import (
    DustParticleSI,
    MaxwellianPlasmaSI,
    OpenDustJAX,
    PlaneDiagnosticConfig,
    PolynomialBackgroundConfig,
    SimulationConfigSI,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particles", type=int, default=2**16)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--dt", type=float, default=3.5148240854e-9)
    parser.add_argument("--seed", type=int, default=2017)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--grid-nx", type=int, default=121)
    parser.add_argument("--grid-nz", type=int, default=121)
    parser.add_argument(
        "--slice-thickness-debye",
        type=float,
        default=0.2,
        help="Thickness of the density-deposition slab in Debye lengths.",
    )
    parser.add_argument("--diagnostic-start-step", type=int, default=1000)
    parser.add_argument("--diagnostic-interval", type=int, default=100)
    parser.add_argument("--diagnostic-source-block-size", type=int, default=4096)
    parser.add_argument("--background-samples", type=int, default=1_357_168)
    parser.add_argument("--background-source-block-size", type=int, default=4096)
    parser.add_argument(
        "--background-cache-dir",
        type=Path,
        default=REPO_ROOT
        / "performance/JAX/results/piel2017_figure1_consistent/background_cache",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT
        / "performance/JAX/results/piel2017_figure1_consistent/diagnostic_run",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if jax.default_backend() != "gpu":
        raise SystemExit("This simulation requires the JAX CUDA backend")
    if args.particles <= 0 or args.steps <= 0 or args.dt <= 0.0:
        raise SystemExit("particles, steps, and dt must be positive")
    if args.grid_nx < 2 or args.grid_nz < 2:
        raise SystemExit("grid dimensions must be at least 2")
    if args.slice_thickness_debye <= 0.0:
        raise SystemExit("slice-thickness-debye must be positive")
    if not 0 <= args.diagnostic_start_step <= args.steps:
        raise SystemExit("diagnostic-start-step must lie between 0 and steps")
    if args.diagnostic_interval <= 0:
        raise SystemExit("diagnostic-interval must be positive")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    plasma = MaxwellianPlasmaSI(
        ion_temperature_K=290.11,
        electron_temperature_K=29_011.0,
        density_m3=1.0e14,
        mach=1.0,
        ion_mass_kg=6.6335209e-26,
    )
    screening_length_m = plasma.screening_length_m
    simulation = SimulationConfigSI(
        radius_m=3.0 * screening_length_m,
        height_m=6.0 * screening_length_m,
        particle_count=args.particles,
        steps=args.steps,
        dt_s=args.dt,
        seed=args.seed,
        force_backend="cuda",
        step_chunk_size=args.chunk_size,
    )
    dust = DustParticleSI(
        position_m=(0.0, 0.0, -screening_length_m),
        radius_m=58.8e-6,
        initial_charge_C=0.0,
        charge_mode="self_consistent",
    )
    background = PolynomialBackgroundConfig(
        enabled=True,
        seed=args.seed,
        sample_count=args.background_samples,
        source_block_size=args.background_source_block_size,
        cache_dir=args.background_cache_dir,
    )
    diagnostics = PlaneDiagnosticConfig(
        nx=args.grid_nx,
        nz=args.grid_nz,
        slice_thickness_m=args.slice_thickness_debye * screening_length_m,
        start_step=args.diagnostic_start_step,
        interval=args.diagnostic_interval,
        output_prefix=output_dir / "piel2017_plane",
        source_block_size=args.diagnostic_source_block_size,
    )

    setup_started = time.perf_counter()
    engine = OpenDustJAX(
        plasma,
        simulation,
        [dust],
        background=background,
        diagnostics=diagnostics,
    )
    setup_s = time.perf_counter() - setup_started

    run_started = time.perf_counter()
    result = engine.run()
    result.final_state.positions_m.block_until_ready()
    run_s = time.perf_counter() - run_started

    history_path = output_dir / "piel2017_dust_history.npz"
    np.savez_compressed(
        history_path,
        time_s=np.asarray(result.times_s, dtype=np.float64),
        charge_C=np.asarray(result.dust_charges_C[:, 0], dtype=np.float64),
        orbit_force_N=np.asarray(
            result.dust_orbit_forces_N[:, 0, :], dtype=np.float64
        ),
        collection_force_N=np.asarray(
            result.dust_collection_forces_N[:, 0, :], dtype=np.float64
        ),
    )

    first_sample = max(args.diagnostic_start_step, 1)
    remainder = (
        first_sample - args.diagnostic_start_step
    ) % args.diagnostic_interval
    if remainder:
        first_sample += args.diagnostic_interval - remainder
    expected_samples = (
        (args.steps - first_sample) // args.diagnostic_interval + 1
        if first_sample <= args.steps
        else 0
    )
    fields_path = result.diagnostic_files[0]
    with np.load(fields_path) as fields:
        actual_samples = int(fields["sample_count"])
        sample_steps = np.asarray(fields["sample_steps"], dtype=np.int64)
    if actual_samples != expected_samples:
        raise RuntimeError(
            f"expected {expected_samples} diagnostic samples, got {actual_samples}"
        )

    summary = {
        "setup_s": setup_s,
        "run_s": run_s,
        "seconds_per_step": run_s / args.steps,
        "particle_count": args.particles,
        "steps": args.steps,
        "dt_s": args.dt,
        "grid_nx": args.grid_nx,
        "grid_nz": args.grid_nz,
        "slice_thickness_m": diagnostics.slice_thickness_m,
        "sample_count": actual_samples,
        "sample_steps": sample_steps.tolist(),
        "screening_length_m": screening_length_m,
        "background_relative_rms_error": engine.background.relative_rms_error,
        "coincident_pair_count": result.coincident_pair_count,
        "diagnostic_files": [str(path) for path in result.diagnostic_files],
        "dust_history_file": str(history_path),
        "gpu": jax.devices()[0].device_kind,
        "jax": jax.__version__,
    }
    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    print("\nPiel2017 OpenDustJAX simulation completed")
    print(f"GPU: {summary['gpu']}")
    print(f"Setup: {setup_s:.3f} s")
    print(f"Simulation: {run_s:.3f} s ({run_s / args.steps:.6e} s/step)")
    print(f"Diagnostic samples: {actual_samples} at steps {sample_steps.tolist()}")
    print(f"Averaged fields: {result.diagnostic_files[0]}")
    print(f"Field metadata: {result.diagnostic_files[1]}")
    print(f"Dust history: {history_path}")
    print(f"Run summary: {summary_path}")


if __name__ == "__main__":
    main()
