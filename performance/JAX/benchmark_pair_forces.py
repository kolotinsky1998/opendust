#!/usr/bin/env python3
"""Compare the direct JAX and OpenMM pair-force implementations on one GPU."""

from __future__ import annotations

import argparse
import csv
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    import jaxlib
except ImportError as exc:  # pragma: no cover - exercised on the benchmark host
    raise SystemExit("JAX with CUDA support is required: install jax[cuda12].") from exc

try:
    import openmm
    from openmm import unit
except ImportError as exc:  # pragma: no cover - exercised on the benchmark host
    raise SystemExit("OpenMM with its CUDA platform is required.") from exc

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from opendust_jax.pair_forces import softened_yukawa_forces


@dataclass(frozen=True)
class Measurement:
    implementation: str
    n_particles: int
    block_size: int | None
    compile_or_setup_s: float
    median_s: float
    minimum_s: float
    p90_s: float
    ordered_pairs_per_s: float
    ns_per_ordered_pair: float
    process_gpu_memory_mib: float | None


@dataclass(frozen=True)
class Validation:
    n_particles: int
    relative_l2: float
    max_absolute_error: float
    max_relative_error: float
    all_finite: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark exact softened-Yukawa forces in JAX and OpenMM."
    )
    parser.add_argument("--powers", type=int, nargs="+", default=[15, 16, 17, 18])
    parser.add_argument("--block-sizes", type=int, nargs="+", default=[32, 64, 128, 256])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--device-index", default="0")
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    parser.add_argument(
        "--allow-non-cuda",
        action="store_true",
        help="Allow a diagnostic run even when JAX or OpenMM is not using CUDA.",
    )
    args = parser.parse_args()
    if any(power <= 0 for power in args.powers):
        parser.error("all powers must be positive")
    if any(size <= 0 for size in args.block_sizes):
        parser.error("all block sizes must be positive")
    if args.warmup < 1 or args.repeat < 1:
        parser.error("warmup and repeat must both be positive")
    return args


def sample_cylinder(
    n_particles: int, *, radius: float = 3.0, height: float = 6.0, seed: int
) -> np.ndarray:
    """Generate deterministic host coordinates shared by both implementations."""

    rng = np.random.default_rng(seed)
    radial = radius * np.sqrt(rng.random(n_particles, dtype=np.float32))
    angle = 2.0 * np.pi * rng.random(n_particles, dtype=np.float32)
    z = height * (rng.random(n_particles, dtype=np.float32) - 0.5)
    return np.column_stack(
        (radial * np.cos(angle), radial * np.sin(angle), z)
    ).astype(np.float32, copy=False)


def percentile_90(samples: list[float]) -> float:
    return float(np.percentile(np.asarray(samples), 90.0))


def process_gpu_memory_mib() -> float | None:
    """Return retained GPU memory for this PID when nvidia-smi exposes it."""

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None

    total = 0.0
    found = False
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            continue
        try:
            if int(fields[0]) == os.getpid():
                total += float(fields[1])
                found = True
        except ValueError:
            continue
    return total if found else None


def measure(
    operation: Callable[[], np.ndarray], warmup: int, repeat: int
) -> tuple[list[float], np.ndarray]:
    result = np.empty((0, 3), dtype=np.float32)
    for _ in range(warmup):
        result = operation()

    samples: list[float] = []
    for _ in range(repeat):
        start = time.perf_counter()
        result = operation()
        samples.append(time.perf_counter() - start)
    return samples, result


def build_openmm_context(
    n_particles: int,
    *,
    prefactor: float,
    screening_length: float,
    softening: float,
    device_index: str,
    platform_name: str,
) -> tuple[openmm.Context, openmm.Integrator, float]:
    setup_start = time.perf_counter()
    system = openmm.System()
    force = openmm.CustomNonbondedForce(
        "prefactor*exp(-r/screening_length)/(r+softening)"
    )
    force.addGlobalParameter("prefactor", prefactor)
    force.addGlobalParameter("screening_length", screening_length)
    force.addGlobalParameter("softening", softening)
    for _ in range(n_particles):
        system.addParticle(1.0)
        force.addParticle([])
    system.addForce(force)
    integrator = openmm.VerletIntegrator(1.0)
    selected_platform = openmm.Platform.getPlatformByName(platform_name)
    properties = (
        {"DeviceIndex": device_index, "Precision": "single"}
        if platform_name == "CUDA"
        else {}
    )
    context = openmm.Context(system, integrator, selected_platform, properties)
    return context, integrator, time.perf_counter() - setup_start


def openmm_force_call(context: openmm.Context, positions: np.ndarray) -> np.ndarray:
    # setPositions invalidates cached forces and makes host-to-device transfer
    # symmetric with the JAX end-to-end operation below.
    context.setPositions(positions)
    state = context.getState(getForces=True)
    return np.asarray(
        state.getForces(asNumpy=True).value_in_unit(
            unit.kilojoule_per_mole / unit.nanometer
        ),
        dtype=np.float32,
    )


def jax_force_call(
    positions: np.ndarray,
    *,
    prefactor: float,
    screening_length: float,
    softening: float,
    block_size: int,
) -> np.ndarray:
    # device_put plus np.asarray makes both transfer directions part of the
    # timed end-to-end call. np.asarray also provides explicit synchronization.
    device_positions = jax.device_put(positions)
    forces = softened_yukawa_forces(
        device_positions,
        prefactor,
        screening_length,
        softening,
        block_size=block_size,
    )
    forces.block_until_ready()
    return np.asarray(forces)


def measurement_from_samples(
    implementation: str,
    n_particles: int,
    block_size: int | None,
    compile_or_setup_s: float,
    samples: list[float],
) -> Measurement:
    median = statistics.median(samples)
    ordered_pairs = n_particles * (n_particles - 1)
    return Measurement(
        implementation=implementation,
        n_particles=n_particles,
        block_size=block_size,
        compile_or_setup_s=compile_or_setup_s,
        median_s=median,
        minimum_s=min(samples),
        p90_s=percentile_90(samples),
        ordered_pairs_per_s=ordered_pairs / median,
        ns_per_ordered_pair=median * 1.0e9 / ordered_pairs,
        process_gpu_memory_mib=process_gpu_memory_mib(),
    )


def validate_against_openmm(
    positions: np.ndarray,
    context: openmm.Context,
    *,
    prefactor: float,
    screening_length: float,
    softening: float,
) -> Validation:
    expected = openmm_force_call(context, positions).astype(np.float64)
    actual = jax_force_call(
        positions,
        prefactor=prefactor,
        screening_length=screening_length,
        softening=softening,
        block_size=128,
    ).astype(np.float64)
    difference = actual - expected
    expected_norm = np.linalg.norm(expected)
    relative_l2 = np.linalg.norm(difference) / expected_norm
    max_absolute = float(np.max(np.abs(difference)))
    denominator = np.maximum(np.abs(expected), np.finfo(np.float32).eps)
    max_relative = float(np.max(np.abs(difference) / denominator))
    return Validation(
        n_particles=len(positions),
        relative_l2=float(relative_l2),
        max_absolute_error=max_absolute,
        max_relative_error=max_relative,
        all_finite=bool(np.all(np.isfinite(actual))),
    )


def environment_info(openmm_platform_name: str) -> dict[str, str]:
    try:
        from jax.extend import backend as jax_backend

        platform_version = jax_backend.get_backend().platform_version
    except Exception:
        platform_version = "unavailable"

    try:
        smi = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        smi = "unavailable"

    return {
        "python": platform.python_version(),
        "jax": jax.__version__,
        "jaxlib_xla": jaxlib.__version__,
        "jax_backend": jax.default_backend(),
        "jax_platform_version": str(platform_version),
        "openmm": openmm.__version__,
        "openmm_platform": openmm_platform_name,
        "nvidia_smi": smi,
    }


def write_csv(path: Path, measurements: list[Measurement]) -> None:
    rows = [asdict(item) for item in measurements]
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    path: Path,
    info: dict[str, str],
    validation: Validation,
    measurements: list[Measurement],
) -> None:
    openmm_by_n = {
        item.n_particles: item
        for item in measurements
        if item.implementation == "openmm"
    }
    best_jax: dict[int, Measurement] = {}
    for item in measurements:
        if item.implementation != "jax":
            continue
        previous = best_jax.get(item.n_particles)
        if previous is None or item.median_s < previous.median_s:
            best_jax[item.n_particles] = item

    lines = [
        "# JAX/OpenMM direct pair-force benchmark",
        "",
        "## Environment",
        "",
    ]
    lines.extend(f"- {key}: `{value}`" for key, value in info.items())
    lines.extend(
        [
            "",
            "## Correctness",
            "",
            "| N | relative L2 | max abs error | max relative error | finite |",
            "|---:|---:|---:|---:|:---:|",
            f"| {validation.n_particles} | {validation.relative_l2:.6e} | "
            f"{validation.max_absolute_error:.6e} | "
            f"{validation.max_relative_error:.6e} | {validation.all_finite} |",
            "",
            "Required relative-L2 tolerance: `5e-5`.",
            "",
            "## Best JAX result per particle count",
            "",
            "| N | JAX block | JAX median, s | OpenMM median, s | JAX/OpenMM | "
            "JAX ordered pairs/s | JAX ns/pair | time growth |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    previous_jax: Measurement | None = None
    for n_particles in sorted(best_jax):
        jax_item = best_jax[n_particles]
        openmm_item = openmm_by_n[n_particles]
        ratio = jax_item.median_s / openmm_item.median_s
        growth = (
            "-"
            if previous_jax is None
            else f"{jax_item.median_s / previous_jax.median_s:.3f}x"
        )
        lines.append(
            f"| {n_particles} | {jax_item.block_size} | {jax_item.median_s:.6f} | "
            f"{openmm_item.median_s:.6f} | {ratio:.3f} | "
            f"{jax_item.ordered_pairs_per_s:.6e} | "
            f"{jax_item.ns_per_ordered_pair:.6e} | {growth} |"
        )
        previous_jax = jax_item

    ratios = [
        best_jax[n].median_s / openmm_by_n[n].median_s for n in sorted(best_jax)
    ]
    minimum_ratio = min(ratios)
    maximum_ratio = max(ratios)
    conclusion = (
        f"The measured JAX/OpenMM time ratio ranges from {minimum_ratio:.3f} to "
        f"{maximum_ratio:.3f}. No performance pass/fail threshold is applied. "
        "Use this range and an Nsight profile to decide whether to retain pure "
        "XLA or next evaluate symmetric tiling and Pallas."
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            conclusion,
            "",
            "`process_gpu_memory_mib` in the CSV is retained process memory sampled "
            "after each configuration, not a kernel-level peak. Leave it empty when "
            "the NVIDIA process-memory query is unavailable.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def print_summary(measurements: list[Measurement]) -> None:
    print(
        "implementation\tN\tblock\tcompile/setup_s\tmedian_s\tmin_s\t"
        "p90_s\tordered_pairs/s\tprocess_gpu_MiB"
    )
    for item in measurements:
        memory = (
            "n/a"
            if item.process_gpu_memory_mib is None
            else f"{item.process_gpu_memory_mib:.1f}"
        )
        block = "-" if item.block_size is None else str(item.block_size)
        print(
            f"{item.implementation}\t{item.n_particles}\t{block}\t"
            f"{item.compile_or_setup_s:.6f}\t{item.median_s:.6f}\t"
            f"{item.minimum_s:.6f}\t{item.p90_s:.6f}\t"
            f"{item.ordered_pairs_per_s:.6e}\t{memory}"
        )


def main() -> None:
    args = parse_args()
    try:
        openmm.Platform.getPlatformByName("CUDA")
        openmm_platform_name = "CUDA"
    except Exception:
        if not args.allow_non_cuda:
            raise SystemExit("The OpenMM CUDA platform is unavailable.")
        openmm_platform_name = "CPU"

    info = environment_info(openmm_platform_name)
    print("Environment:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    if not args.allow_non_cuda and jax.default_backend() != "gpu":
        raise SystemExit(
            "JAX is not using a GPU. Install CUDA-enabled JAX or pass "
            "--allow-non-cuda for a diagnostic-only run."
        )

    prefactor = 1.0
    screening_length = 1.0
    softening = 1.0e-3
    args.output_directory.mkdir(parents=True, exist_ok=True)

    validation_positions = sample_cylinder(1024, seed=12345)
    validation_context, validation_integrator, _ = build_openmm_context(
        len(validation_positions),
        prefactor=prefactor,
        screening_length=screening_length,
        softening=softening,
        device_index=args.device_index,
        platform_name=openmm_platform_name,
    )
    validation = validate_against_openmm(
        validation_positions,
        validation_context,
        prefactor=prefactor,
        screening_length=screening_length,
        softening=softening,
    )
    del validation_context, validation_integrator
    print(f"Validation: {validation}")
    if not validation.all_finite or validation.relative_l2 > 5.0e-5:
        raise SystemExit("Correctness validation failed; benchmark aborted.")

    measurements: list[Measurement] = []
    for power in args.powers:
        n_particles = 2**power
        positions = sample_cylinder(n_particles, seed=power)

        context, integrator, setup_s = build_openmm_context(
            n_particles,
            prefactor=prefactor,
            screening_length=screening_length,
            softening=softening,
            device_index=args.device_index,
            platform_name=openmm_platform_name,
        )
        openmm_samples, _ = measure(
            lambda: openmm_force_call(context, positions), args.warmup, args.repeat
        )
        measurements.append(
            measurement_from_samples(
                "openmm", n_particles, None, setup_s, openmm_samples
            )
        )
        del context, integrator

        for block_size in args.block_sizes:
            compile_start = time.perf_counter()
            jax_force_call(
                positions,
                prefactor=prefactor,
                screening_length=screening_length,
                softening=softening,
                block_size=block_size,
            )
            compile_s = time.perf_counter() - compile_start
            jax_samples, _ = measure(
                lambda block_size=block_size: jax_force_call(
                    positions,
                    prefactor=prefactor,
                    screening_length=screening_length,
                    softening=softening,
                    block_size=block_size,
                ),
                args.warmup,
                args.repeat,
            )
            measurements.append(
                measurement_from_samples(
                    "jax", n_particles, block_size, compile_s, jax_samples
                )
            )

    csv_path = args.output_directory / "pair_forces_results.csv"
    report_path = args.output_directory / "pair_forces_report.md"
    write_csv(csv_path, measurements)
    write_report(report_path, info, validation, measurements)
    print_summary(measurements)
    print(f"CSV: {csv_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
