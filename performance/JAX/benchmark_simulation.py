#!/usr/bin/env python3
"""Benchmark complete OpenDustJAX integration steps on one NVIDIA GPU."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics
import sys
import time

import jax
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from opendust_jax import (
    DustParticleSI,
    MaxwellianPlasmaSI,
    OpenDustJAX,
    PolynomialBackgroundConfig,
    SimulationConfigSI,
)


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values), q))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--powers", nargs="+", type=int, default=[15, 16, 17, 18])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "performance/JAX/results/simulation_results.csv",
    )
    args = parser.parse_args()
    if jax.default_backend() != "gpu":
        raise SystemExit("This benchmark requires the JAX CUDA backend")
    if args.warmup < 1:
        raise SystemExit("--warmup must be at least 1 to exclude JIT compilation")

    plasma = MaxwellianPlasmaSI(
        ion_temperature_K=290.11,
        electron_temperature_K=29_011.0,
        density_m3=1.0e14,
        mach=1.0,
        ion_mass_kg=6.6335209e-26,
    )
    rows: list[dict] = []
    print(
        "implementation\tN\tsteps\tcompile_warmup_s\tmedian_s\tmin_s\t"
        "p90_s\tseconds/step"
    )
    for power in args.powers:
        count = 2**power
        screening = plasma.screening_length_m
        config = SimulationConfigSI(
            radius_m=3.0 * screening,
            height_m=6.0 * screening,
            particle_count=count,
            steps=args.steps,
            dt_s=3.5148e-10,
            seed=2027,
            force_backend="cuda",
            step_chunk_size=args.chunk_size,
        )
        dust = DustParticleSI(
            position_m=(0.0, 0.0, -screening),
            radius_m=58.8e-6,
            initial_charge_C=-392_500 * 1.602176634e-19,
            charge_mode="fixed",
        )
        engine = OpenDustJAX(
            plasma,
            config,
            [dust],
            background=PolynomialBackgroundConfig(enabled=False),
        )
        state = engine.initial_state()
        state.positions_m.block_until_ready()
        warmup_timings = []
        for _ in range(args.warmup):
            warmup_start = time.perf_counter()
            warmed = engine.run(state).final_state
            warmed.positions_m.block_until_ready()
            warmup_timings.append(time.perf_counter() - warmup_start)
        timings = []
        for _ in range(args.repeat):
            start = time.perf_counter()
            result = engine.run(state)
            result.final_state.positions_m.block_until_ready()
            timings.append(time.perf_counter() - start)
        row = {
            "implementation": "opendust-jax-cuda",
            "N": count,
            "steps": args.steps,
            "compile_warmup_s": warmup_timings[0],
            "median_s": statistics.median(timings),
            "min_s": min(timings),
            "p90_s": percentile(timings, 90),
        }
        row["seconds_per_step"] = row["median_s"] / args.steps
        rows.append(row)
        print(
            f"opendust-jax-cuda\t{count}\t{args.steps}\t"
            f"{row['compile_warmup_s']:.6f}\t{row['median_s']:.6f}\t"
            f"{row['min_s']:.6f}\t{row['p90_s']:.6f}\t{row['seconds_per_step']:.6e}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV: {args.output}")


if __name__ == "__main__":
    main()
