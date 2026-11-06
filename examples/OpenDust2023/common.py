"""Shared execution, persistence, and plotting helpers for OpenDust 2023 figures."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass
import fcntl
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
from typing import Callable, Sequence

import numpy as np


ELEMENTARY_CHARGE_C = 1.602176634e-19
DEFAULT_SEED = 2017
DEFAULT_BACKGROUND_SAMPLES = 1_357_168


@dataclass(frozen=True)
class CaseSetup:
    plasma: object
    simulation: object
    dust_particles: Sequence[object]
    background: object
    case_metadata: dict
    average_start_step: int


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def add_source_tree_to_path() -> None:
    source = repository_root() / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))


def common_parser(description: str, case_count: int) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--case-index",
        type=int,
        default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")),
        choices=range(case_count),
    )
    parser.add_argument("--particles", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--background-samples", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def production_or_override(value: int | None, production: int) -> int:
    return production if value is None else value


def averaging_start(requested: int, steps: int) -> int:
    """Preserve production averaging and provide a useful window for smoke runs."""
    return requested if steps > requested else max(0, steps // 2)


@contextmanager
def exclusive_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _jsonable(value):
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, suffix=".json", delete=False
    ) as stream:
        json.dump(_jsonable(value), stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_csv(path: Path, table: np.ndarray, header: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, suffix=".csv", delete=False
    ) as stream:
        np.savetxt(stream, table, delimiter=",", header=header, comments="", fmt="%.12e")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _history_table(result) -> tuple[np.ndarray, str]:
    times = np.asarray(result.times_s, dtype=np.float64)
    charges = np.asarray(result.dust_charges_C, dtype=np.float64)
    components = (
        ("orbit", np.asarray(result.dust_orbit_forces_N, dtype=np.float64)),
        ("collection", np.asarray(result.dust_collection_forces_N, dtype=np.float64)),
        ("dust", np.asarray(result.dust_dust_forces_N, dtype=np.float64)),
        ("external", np.asarray(result.dust_external_forces_N, dtype=np.float64)),
    )
    columns = [times]
    names = ["time_s"]
    for dust in range(charges.shape[1]):
        columns.append(charges[:, dust])
        names.append(f"dust{dust}_charge_C")
        for label, values in components:
            for axis, axis_name in enumerate("xyz"):
                columns.append(values[:, dust, axis])
                names.append(f"dust{dust}_{label}_{axis_name}_N")
    return np.column_stack(columns), ",".join(names)


def save_case_result(
    case_dir: Path,
    setup: CaseSetup,
    result,
    timing: dict,
    resolved_background: dict | None = None,
) -> dict:
    times = np.asarray(result.times_s, dtype=np.float64)
    charges = np.asarray(result.dust_charges_C, dtype=np.float64)
    orbit = np.asarray(result.dust_orbit_forces_N, dtype=np.float64)
    collection = np.asarray(result.dust_collection_forces_N, dtype=np.float64)
    dust = np.asarray(result.dust_dust_forces_N, dtype=np.float64)
    external = np.asarray(result.dust_external_forces_N, dtype=np.float64)
    steps = np.arange(times.size, dtype=np.int64)
    mask = steps > setup.average_start_step
    if not np.any(mask):
        raise RuntimeError("the configured averaging interval contains no completed steps")

    atomic_npz(
        case_dir / "history.npz",
        step=steps,
        time_s=times,
        charge_C=charges,
        orbit_force_N=orbit,
        collection_force_N=collection,
        dust_force_N=dust,
        external_force_N=external,
    )
    table, header = _history_table(result)
    atomic_csv(case_dir / "history.csv", table, header)

    summary = {
        "case": setup.case_metadata,
        "average_start_step_exclusive": setup.average_start_step,
        "average_sample_count": int(np.count_nonzero(mask)),
        "mean_charge_C": np.mean(charges[mask], axis=0),
        "mean_orbit_force_N": np.mean(orbit[mask], axis=0),
        "mean_collection_force_N": np.mean(collection[mask], axis=0),
        "mean_dust_force_N": np.mean(dust[mask], axis=0),
        "mean_external_force_N": np.mean(external[mask], axis=0),
        "coincident_pair_count": result.coincident_pair_count,
        "plasma": setup.plasma,
        "simulation": setup.simulation,
        "dust_particles": list(setup.dust_particles),
        "background": setup.background,
        "resolved_background": resolved_background,
        "timing": timing,
    }
    atomic_json(case_dir / "summary.json", summary)
    return _jsonable(summary)


def run_case_cli(
    figure_dir: Path,
    case_count: int,
    description: str,
    build_case: Callable[[argparse.Namespace, Path], CaseSetup],
) -> None:
    parser = common_parser(description, case_count)
    args = parser.parse_args()
    if args.particles is not None and args.particles <= 0:
        parser.error("--particles must be positive")
    if args.steps is not None and args.steps <= 0:
        parser.error("--steps must be positive")
    if args.background_samples is not None and args.background_samples < 256:
        parser.error("--background-samples must be at least 256")
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")

    add_source_tree_to_path()
    import jax
    from jax.extend import backend as jax_backend
    from opendust_jax import OpenDustJAX, cuda_pair_forces_available

    if jax.default_backend() != "gpu":
        raise SystemExit("This production example requires the JAX CUDA backend")
    if not cuda_pair_forces_available():
        raise SystemExit(
            "The OpenDustJAX CUDA FFI library is unavailable. Build it with "
            "scripts/build_cuda_pair_forces.py before submitting the examples."
        )

    results_dir = (args.output_dir or (figure_dir / "results")).resolve()
    run_id = (
        os.environ.get("SLURM_ARRAY_JOB_ID")
        or os.environ.get("SLURM_JOB_ID")
        or "manual"
    )
    case_dir = results_dir / f"case_{args.case_index:02d}"
    case_dir.mkdir(parents=True, exist_ok=True)
    marker = case_dir / ".done.json"
    marker.unlink(missing_ok=True)

    setup = build_case(args, results_dir)
    setup_started = time.perf_counter()
    with exclusive_lock(results_dir / "background_cache.lock"):
        engine = OpenDustJAX(
            setup.plasma,
            setup.simulation,
            setup.dust_particles,
            background=setup.background,
        )
    setup_s = time.perf_counter() - setup_started

    run_started = time.perf_counter()
    result = engine.run()
    result.final_state.positions_m.block_until_ready()
    run_s = time.perf_counter() - run_started
    timing = {
        "setup_s": setup_s,
        "integration_s": run_s,
        "seconds_per_step": run_s / setup.simulation.steps,
        "python": platform.python_version(),
        "jax": jax.__version__,
        "jax_backend": jax.default_backend(),
        "jax_platform_version": jax_backend.get_backend().platform_version,
        "gpu": jax.devices()[0].device_kind,
    }
    summary = save_case_result(
        case_dir,
        setup,
        result,
        timing,
        resolved_background={
            "coefficients": engine.background.coefficients,
            "relative_rms_error": engine.background.relative_rms_error,
            "sample_count": engine.background.sample_count,
            "cache_key": engine.background.cache_key,
        },
    )
    atomic_json(
        marker,
        {"case_index": args.case_index, "completed": True, "run_id": run_id},
    )
    print(json.dumps(summary, indent=2, sort_keys=True))

    plot_command = [
        sys.executable,
        str(figure_dir / "plot.py"),
        "--if-complete",
        "--output-dir",
        str(results_dir),
        "--run-id",
        run_id,
    ]
    subprocess.run(plot_command, check=True)


def load_case(results_dir: Path, index: int) -> tuple[dict, dict[str, np.ndarray]]:
    case_dir = results_dir / f"case_{index:02d}"
    summary = json.loads((case_dir / "summary.json").read_text(encoding="utf-8"))
    with np.load(case_dir / "history.npz") as data:
        history = {name: np.asarray(data[name]) for name in data.files}
    return summary, history


def run_plot_cli(
    figure_dir: Path,
    expected_cases: int,
    generate: Callable[[Path], None],
) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--if-complete", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=figure_dir / "results")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    results_dir = args.output_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(results_dir / ".plot.lock"):
        missing = []
        for index in range(expected_cases):
            marker = results_dir / f"case_{index:02d}" / ".done.json"
            if not marker.is_file():
                missing.append(index)
                continue
            if args.run_id is not None:
                payload = json.loads(marker.read_text(encoding="utf-8"))
                if payload.get("run_id") != args.run_id:
                    missing.append(index)
        if missing:
            message = f"Results are incomplete; missing cases: {missing}"
            if args.if_complete:
                print(message)
                return
            raise SystemExit(message)
        generate(results_dir)
        print(f"Final table and plots written to {results_dir}")


def save_figure(fig, results_dir: Path, stem: str) -> None:
    for suffix in ("png", "pdf"):
        final = results_dir / f"{stem}.{suffix}"
        with tempfile.NamedTemporaryFile(
            dir=results_dir, suffix=f".{suffix}", delete=False
        ) as stream:
            temporary = Path(stream.name)
        try:
            fig.savefig(temporary, dpi=200, bbox_inches="tight")
            os.replace(temporary, final)
        finally:
            temporary.unlink(missing_ok=True)
