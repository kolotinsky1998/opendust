#!/usr/bin/env python3
"""Controlled OpenMM/OpenDustJAX comparison for Piel2017 Figure 1.

The two implementations use the same initial ions and the same polynomial
background.  Reinjection RNGs differ, so long-time charge and force statistics
are compared instead of pointwise trajectories.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

T_I_K = 290.11
T_E_K = 29_011.0
DENSITY_M3 = 1.0e14
ION_MASS_KG = 6.6335209e-26
MACH = 1.0
DUST_RADIUS_M = 58.8e-6


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--backend",
        choices=("all", "prepare", "jax", "openmm", "compare"),
        default="all",
    )
    result.add_argument("--particles", type=int, default=2**16)
    result.add_argument("--steps", type=int, default=3000)
    result.add_argument("--dt", type=float, default=3.5148240854e-9)
    result.add_argument("--seed", type=int, default=2017)
    result.add_argument("--chunk-size", type=int, default=100)
    result.add_argument("--average-start-step", type=int, default=1001)
    result.add_argument(
        "--background-samples",
        type=int,
        default=1_357_168,
        help="Legacy example-equivalent source count; use 65536 for a quick run.",
    )
    result.add_argument("--background-source-block-size", type=int, default=4096)
    result.add_argument("--device-index", default="0")
    result.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT
        / "performance/JAX/results/piel2017_figure1_consistent",
    )
    return result


def common_paths(args: argparse.Namespace) -> dict[str, Path]:
    output = args.output_dir.resolve()
    return {
        "output": output,
        "shared": output / "shared_inputs.npz",
        "preparation": output / "preparation.json",
        "jax": output / "jax_results.npz",
        "jax_timing": output / "jax_timing.json",
        "openmm": output / "openmm_results.npz",
        "openmm_timing": output / "openmm_timing.json",
        "restart": output / "openmm_initial_state.tsv",
        "summary": output / "comparison.json",
        "report": output / "comparison.md",
    }


def configurations(args: argparse.Namespace, force_backend: str):
    from opendust_jax import (
        DustParticleSI,
        MaxwellianPlasmaSI,
        PolynomialBackgroundConfig,
        SimulationConfigSI,
    )

    plasma = MaxwellianPlasmaSI(
        ion_temperature_K=T_I_K,
        electron_temperature_K=T_E_K,
        density_m3=DENSITY_M3,
        mach=MACH,
        ion_mass_kg=ION_MASS_KG,
    )
    simulation = SimulationConfigSI(
        radius_m=3.0 * plasma.screening_length_m,
        height_m=6.0 * plasma.screening_length_m,
        particle_count=args.particles,
        steps=args.steps,
        dt_s=args.dt,
        seed=args.seed,
        force_backend=force_backend,
        step_chunk_size=args.chunk_size,
    )
    dust = DustParticleSI(
        position_m=(0.0, 0.0, -plasma.screening_length_m),
        radius_m=DUST_RADIUS_M,
        initial_charge_C=0.0,
        charge_mode="self_consistent",
    )
    background = PolynomialBackgroundConfig(
        enabled=True,
        seed=args.seed,
        sample_count=args.background_samples,
        source_block_size=args.background_source_block_size,
        cache_dir=args.output_dir / "background_cache",
    )
    return plasma, simulation, dust, background


def prepare(args: argparse.Namespace) -> None:
    from opendust_jax import OpenDustJAX

    paths = common_paths(args)
    paths["output"].mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    plasma, simulation, dust, background = configurations(args, "jax")
    engine = OpenDustJAX(plasma, simulation, [dust], background=background)
    positions, velocities = engine._initial_arrays()
    elapsed = time.perf_counter() - started
    np.savez_compressed(
        paths["shared"],
        positions_m=positions,
        velocities_m_s=velocities,
        background_coefficients=engine.background.coefficients,
        background_relative_rms_error=np.asarray(
            engine.background.relative_rms_error
        ),
    )
    metadata = {
        "preparation_s": elapsed,
        "particles": args.particles,
        "steps": args.steps,
        "dt_s": args.dt,
        "seed": args.seed,
        "background_samples": args.background_samples,
        "background_relative_rms_error": engine.background.relative_rms_error,
        "screening_length_m": plasma.screening_length_m,
    }
    paths["preparation"].write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Prepared common initial state and background in {elapsed:.3f} s")
    print(f"Shared inputs: {paths['shared']}")


def verify_shared_background(engine, shared: np.lib.npyio.NpzFile) -> None:
    expected = np.asarray(shared["background_coefficients"], dtype=np.float64)
    actual = np.asarray(engine.background.coefficients, dtype=np.float64)
    if not np.array_equal(expected, actual):
        raise RuntimeError("cached JAX background differs from shared coefficients")


def save_series(
    path: Path,
    steps: np.ndarray,
    dt_s: float,
    charge_C: np.ndarray,
    orbit_N: np.ndarray,
    collection_N: np.ndarray,
) -> None:
    steps = np.asarray(steps, dtype=np.int64)
    charge_C = np.asarray(charge_C, dtype=np.float64)
    orbit_N = np.asarray(orbit_N, dtype=np.float64)
    collection_N = np.asarray(collection_N, dtype=np.float64)
    total_N = orbit_N + collection_N
    np.savez_compressed(
        path,
        step=steps,
        time_s=np.asarray(steps, dtype=np.float64) * dt_s,
        charge_C=charge_C,
        orbit_force_N=orbit_N,
        collection_force_N=collection_N,
        total_ion_force_N=total_N,
    )
    table = np.column_stack(
        (
            steps,
            steps.astype(np.float64) * dt_s,
            charge_C,
            orbit_N,
            collection_N,
            total_N,
        )
    )
    np.savetxt(
        path.with_suffix(".csv"),
        table,
        delimiter=",",
        header=(
            "step,time_s,charge_C,orbit_x_N,orbit_y_N,orbit_z_N,"
            "collection_x_N,collection_y_N,collection_z_N,"
            "total_x_N,total_y_N,total_z_N"
        ),
        comments="",
        fmt=["%.0f"] + ["%.12e"] * 11,
    )


def run_jax(args: argparse.Namespace) -> None:
    import jax
    import jax.numpy as jnp

    from opendust_jax import OpenDustJAX, SimulationState

    if jax.default_backend() != "gpu":
        raise SystemExit("The JAX Piel2017 benchmark requires a CUDA GPU")
    paths = common_paths(args)
    shared = np.load(paths["shared"])
    plasma, simulation, dust, background = configurations(args, "cuda")

    setup_started = time.perf_counter()
    engine = OpenDustJAX(plasma, simulation, [dust], background=background)
    verify_shared_background(engine, shared)
    positions = jnp.asarray(shared["positions_m"], dtype=jnp.float32)
    velocities = jnp.asarray(shared["velocities_m_s"], dtype=jnp.float32)
    charges = jnp.zeros((1,), dtype=jnp.float32)
    initial_force, _, _ = engine._force(positions, charges)
    initial_force.block_until_ready()
    state = SimulationState(
        positions,
        velocities,
        initial_force,
        charges,
        jnp.asarray(0, jnp.int32),
        jax.random.PRNGKey(args.seed),
    )
    setup_s = time.perf_counter() - setup_started

    compile_started = time.perf_counter()
    warmed = engine.run(state)
    warmed.final_state.positions_m.block_until_ready()
    compile_warmup_s = time.perf_counter() - compile_started

    integration_started = time.perf_counter()
    result = engine.run(state)
    result.final_state.positions_m.block_until_ready()
    integration_s = time.perf_counter() - integration_started

    completed_steps = np.arange(1, args.steps + 1, dtype=np.int64)
    save_series(
        paths["jax"],
        completed_steps,
        args.dt,
        np.asarray(result.dust_charges_C[1:, 0]),
        np.asarray(result.dust_orbit_forces_N[1:, 0, :]),
        np.asarray(result.dust_collection_forces_N[1:, 0, :]),
    )
    timing = {
        "backend": "opendust-jax-cuda",
        "setup_s": setup_s,
        "compile_warmup_s": compile_warmup_s,
        "integration_s": integration_s,
        "seconds_per_step": integration_s / args.steps,
        "jax": jax.__version__,
        "jax_platform": jax.devices()[0].device_kind,
    }
    paths["jax_timing"].write_text(
        json.dumps(timing, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(timing, indent=2, sort_keys=True))


def run_openmm(args: argparse.Namespace) -> None:
    import openmm

    try:
        from opendust.opendust import (
            DustParticle,
            OpenDust,
            OutputParameters,
            PlasmaParametersInSIUnitsMaxwell,
            SimulationParametersInSIUnits,
        )
    except ModuleNotFoundError as error:
        if error.name == "unyt":
            raise SystemExit(
                "The legacy OpenMM implementation requires 'unyt'. Install it "
                "in this environment with: conda install -c conda-forge unyt"
            ) from error
        raise

    paths = common_paths(args)
    shared = np.load(paths["shared"])
    restart_values = np.column_stack(
        (
            np.asarray(shared["positions_m"]) * 1.0e9,
            np.asarray(shared["velocities_m_s"]) * 1.0e-3,
        )
    )
    np.savetxt(paths["restart"], restart_values, delimiter="\t", fmt="%.9e")

    setup_started = time.perf_counter()
    plasma = PlasmaParametersInSIUnitsMaxwell(
        T_I_K, T_E_K, DENSITY_M3, MACH, ION_MASS_KG
    )
    simulation = SimulationParametersInSIUnits(
        3.0 * plasma.r_D_e,
        6.0 * plasma.r_D_e,
        args.particles,
        args.steps,
        args.dt,
        plasma,
    )
    output = OutputParameters(args.steps + 1, args.steps + 1, "", "", "")
    dust = DustParticle(
        0.0, 0.0, -plasma.r_D_e, DUST_RADIUS_M, "consistent", 0.0
    )
    engine = OpenDust(plasma, simulation, output, [dust], "Maxwellian")
    object_setup_s = time.perf_counter() - setup_started
    engine.simulate(
        deviceIndex=args.device_index,
        cutOff=False,
        toRestartFileName=str(paths["restart"]),
        backgroundCoefficients=np.asarray(shared["background_coefficients"]),
        randomSeed=args.seed,
    )

    completed_steps = np.arange(1, args.steps + 1, dtype=np.int64)
    legacy_dust = engine.dustParticles[0]
    save_series(
        paths["openmm"],
        completed_steps,
        args.dt,
        legacy_dust.q,
        legacy_dust.forceIonsOrbit,
        legacy_dust.forceIonsCollect,
    )
    timing = {
        "backend": "legacy-openmm",
        "object_setup_s": object_setup_s,
        "simulate_total_s": engine.simulateTimeSeconds,
        "integration_s": engine.integrationTimeSeconds,
        "seconds_per_step": engine.integrationTimeSeconds / args.steps,
        "openmm": openmm.__version__,
        "openmm_platform": "CUDA",
    }
    paths["openmm_timing"].write_text(
        json.dumps(timing, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(timing, indent=2, sort_keys=True))


def scalar_statistics(values: np.ndarray, start: int) -> dict[str, float]:
    selected = np.asarray(values[start:], dtype=np.float64)
    return {
        "mean": float(np.mean(selected)),
        "std": float(np.std(selected)),
        "final": float(values[-1]),
    }


def relative_difference(left: float, right: float) -> float:
    scale = max(abs(left), abs(right), np.finfo(np.float64).tiny)
    return abs(left - right) / scale


def compare(args: argparse.Namespace) -> None:
    paths = common_paths(args)
    jax_data = np.load(paths["jax"])
    openmm_data = np.load(paths["openmm"])
    jax_timing = json.loads(paths["jax_timing"].read_text(encoding="utf-8"))
    openmm_timing = json.loads(
        paths["openmm_timing"].read_text(encoding="utf-8")
    )
    preparation = json.loads(paths["preparation"].read_text(encoding="utf-8"))
    start = max(args.average_start_step - 1, 0)
    if start >= args.steps:
        raise ValueError("average-start-step must not exceed the step count")

    fields = {
        "charge_C": (jax_data["charge_C"], openmm_data["charge_C"]),
        "orbit_force_z_N": (
            jax_data["orbit_force_N"][:, 2],
            openmm_data["orbit_force_N"][:, 2],
        ),
        "collection_force_z_N": (
            jax_data["collection_force_N"][:, 2],
            openmm_data["collection_force_N"][:, 2],
        ),
        "total_ion_force_z_N": (
            jax_data["total_ion_force_N"][:, 2],
            openmm_data["total_ion_force_N"][:, 2],
        ),
    }
    statistics = {}
    for name, (jax_values, openmm_values) in fields.items():
        jax_stats = scalar_statistics(jax_values, start)
        openmm_stats = scalar_statistics(openmm_values, start)
        statistics[name] = {
            "jax": jax_stats,
            "openmm": openmm_stats,
            "relative_mean_difference": relative_difference(
                jax_stats["mean"], openmm_stats["mean"]
            ),
        }
    speedup = openmm_timing["integration_s"] / jax_timing["integration_s"]
    jax_cold_run_s = jax_timing["setup_s"] + jax_timing["compile_warmup_s"]
    openmm_cold_run_s = (
        openmm_timing["object_setup_s"] + openmm_timing["simulate_total_s"]
    )
    summary = {
        "scenario": "Piel2017/Figure1/consistent",
        "particles": args.particles,
        "steps": args.steps,
        "dt_s": args.dt,
        "average_start_step": args.average_start_step,
        "common_preparation": preparation,
        "timing": {
            "jax": jax_timing,
            "openmm": openmm_timing,
            "openmm_over_jax_integration_speedup": speedup,
            "jax_cold_single_run_s": jax_cold_run_s,
            "openmm_cold_single_run_s": openmm_cold_run_s,
            "openmm_over_jax_cold_run_speedup": (
                openmm_cold_run_s / jax_cold_run_s
            ),
        },
        "statistics": statistics,
        "comparison_note": (
            "Initial ions and polynomial background are identical. Reinjection "
            "RNG streams and corrected JAX charging/indexing rules differ, so "
            "long-time statistics, not pointwise trajectories, are compared."
        ),
    }
    paths["summary"].write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    rows = []
    for name, values in statistics.items():
        rows.append(
            f"| {name} | {values['jax']['mean']:.8e} | "
            f"{values['openmm']['mean']:.8e} | "
            f"{values['relative_mean_difference']:.6e} |"
        )
    report = f"""# Piel2017 Figure 1: OpenDustJAX vs legacy OpenMM

Common initial ion state and polynomial background; averages start at completed
step {args.average_start_step}.

## Timing

| implementation | integration, s | seconds/step | cold single run, s |
| :-- | --: | --: | --: |
| OpenDustJAX CUDA | {jax_timing['integration_s']:.6f} | {jax_timing['seconds_per_step']:.8e} | {jax_cold_run_s:.6f} |
| Legacy OpenMM | {openmm_timing['integration_s']:.6f} | {openmm_timing['seconds_per_step']:.8e} | {openmm_cold_run_s:.6f} |

OpenMM/JAX integration-time ratio: **{speedup:.4f}x**.

JAX setup: {jax_timing['setup_s']:.6f} s. JAX compile plus warm-up:
{jax_timing['compile_warmup_s']:.6f} s. Shared input/background preparation:
{preparation['preparation_s']:.6f} s.

## Charge and axial ion force statistics

| observable | JAX mean | OpenMM mean | relative mean difference |
| :-- | --: | --: | --: |
{os.linesep.join(rows)}

The trajectories are not expected to coincide point by point after stochastic
reinjection. The JAX model also intentionally contains the charging and time-index
corrections documented by the implementation audit.
"""
    paths["report"].write_text(report, encoding="utf-8")
    print(report)
    print(f"JSON: {paths['summary']}")
    print(f"Report: {paths['report']}")


def run_all(args: argparse.Namespace) -> None:
    base = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--particles",
        str(args.particles),
        "--steps",
        str(args.steps),
        "--dt",
        repr(args.dt),
        "--seed",
        str(args.seed),
        "--chunk-size",
        str(args.chunk_size),
        "--average-start-step",
        str(args.average_start_step),
        "--background-samples",
        str(args.background_samples),
        "--background-source-block-size",
        str(args.background_source_block_size),
        "--device-index",
        args.device_index,
        "--output-dir",
        str(args.output_dir),
    ]
    for backend in ("prepare", "jax", "openmm", "compare"):
        print(f"\n=== {backend} ===", flush=True)
        subprocess.run(base + ["--backend", backend], check=True)


def main() -> None:
    args = parser().parse_args()
    if args.particles <= 0 or args.steps <= 0 or args.dt <= 0.0:
        raise SystemExit("particles, steps, and dt must be positive")
    if args.background_samples < 256:
        raise SystemExit("background-samples must be at least 256")
    actions = {
        "all": run_all,
        "prepare": prepare,
        "jax": run_jax,
        "openmm": run_openmm,
        "compare": compare,
    }
    actions[args.backend](args)


if __name__ == "__main__":
    main()
