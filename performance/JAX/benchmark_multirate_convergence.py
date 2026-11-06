#!/usr/bin/env python3
"""Compare three-level time splitting at fixed physical duration."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys

import jax
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from opendust_jax import (  # noqa: E402
    ChargingSubcyclingConfig,
    CollisionSubcyclingConfig,
    DustParticleSI,
    FieldDrivenPlasmaSI,
    OpenDustJAX,
    PlaneDiagnosticConfig,
    PolynomialBackgroundConfig,
    SimulationConfigSI,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dt-ns", nargs="+", type=float, default=[0.5, 1, 2, 5, 10])
    parser.add_argument("--duration-ns", type=float, default=100.0)
    parser.add_argument("--particles", type=int, default=4096)
    parser.add_argument("--backend", choices=("cuda", "jax"), default="cuda")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "performance/JAX/results/multirate_convergence",
    )
    args = parser.parse_args()
    if args.backend == "cuda" and jax.default_backend() != "gpu":
        raise SystemExit("--backend cuda requires the JAX CUDA backend")
    if args.duration_ns <= 0.0 or any(value <= 0.0 for value in args.dt_ns):
        raise SystemExit("time steps and duration must be positive")

    plasma = FieldDrivenPlasmaSI(
        neutral_temperature_K=290.0,
        electron_temperature_K=58_022.0,
        density_m3=1.0e14,
        electric_field_V_m=16_000.0,
        collision_frequency_s=1.0e8,
        ion_mass_kg=6.6335209e-26,
    )
    screening = plasma.screening_length_m
    dust = DustParticleSI(
        (0.0, 0.0, -0.5 * screening),
        4.445e-6,
        initial_charge_C=-2.0e4 * 1.602176634e-19,
        charge_mode="self_consistent",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int]] = []
    wake_profiles: dict[str, np.ndarray] = {}
    z_axis = None

    for dt_ns in args.dt_ns:
        dt_s = dt_ns * 1.0e-9
        steps = max(1, math.ceil(args.duration_ns * 1.0e-9 / dt_s))
        diagnostic_prefix = args.output_dir / f"dt_{dt_ns:g}_ns"
        simulation = SimulationConfigSI(
            radius_m=1.25 * screening,
            height_m=5.0 * screening,
            particle_count=args.particles,
            steps=steps,
            dt_s=dt_s,
            seed=args.seed,
            force_backend=args.backend,
            step_chunk_size=min(steps, 100),
            collision_subcycling=CollisionSubcyclingConfig(),
            charging_subcycling=ChargingSubcyclingConfig(),
        )
        diagnostics = PlaneDiagnosticConfig(
            nx=3,
            nz=81,
            slice_thickness_m=0.1 * screening,
            start_step=steps,
            interval=steps,
            output_prefix=diagnostic_prefix,
        )
        engine = OpenDustJAX(
            plasma,
            simulation,
            [dust],
            background=PolynomialBackgroundConfig(enabled=False),
            diagnostics=diagnostics,
        )
        result = engine.run()
        burn = 1 + steps // 2
        total_force = (
            np.asarray(result.dust_orbit_forces_N)
            + np.asarray(result.dust_collection_forces_N)
            + np.asarray(result.dust_external_forces_N)
        )
        stats = result.collision_statistics
        physical_duration = steps * dt_s
        with np.load(result.diagnostic_files[0], allow_pickle=False) as data:
            z_axis = np.asarray(data["z_m"])
            wake_profiles[f"dt_{dt_ns:g}_ns"] = np.asarray(data["potential_V"])[:, 1]
            density_profile = np.asarray(data["ion_density_m3"])[:, 1]
        velocities = np.asarray(jax.device_get(result.final_state.velocities_m_s))
        rows.append(
            {
                "dt_ii_ns": dt_ns,
                "steps": steps,
                "collision_substeps": engine.collision_substeps,
                "collision_dt_ns": engine.collision_dt_s * 1.0e9,
                "charging_substeps": engine.charging_substeps,
                "integration_segments": engine.integration_segments,
                "mean_density_m3": float(np.mean(density_profile)),
                "mean_vz_m_s": float(np.mean(velocities[:, 2])),
                "collision_rate_s": stats.candidate_count
                / (args.particles * physical_duration),
                "mean_charge_C": float(np.mean(result.dust_charges_C[burn:, 0])),
                "mean_force_z_N": float(np.mean(total_force[burn:, 0, 2])),
                "wake_peak_V": float(np.max(wake_profiles[f"dt_{dt_ns:g}_ns"])),
            }
        )
        print(
            f"dt_ii={dt_ns:g} ns: steps={steps}, Ncoll={engine.collision_substeps}, "
            f"Ncharge={engine.charging_substeps}, segments={engine.integration_segments}"
        )

    csv_path = args.output_dir / "multirate_convergence.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(args.output_dir / "wake_profiles.npz", z_m=z_axis, **wake_profiles)
    (args.output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "plasma": asdict(plasma),
                "dt_ns": args.dt_ns,
                "duration_ns": args.duration_ns,
                "particles": args.particles,
                "seed": args.seed,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
