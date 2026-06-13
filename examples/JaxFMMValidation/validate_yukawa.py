"""Run pure Yukawa direct-vs-FMM validation in a cylindrical source domain."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import jax.numpy as jnp
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
SRC_DIR = REPO_ROOT / "src"
JAXFMM_DIR = REPO_ROOT / "jaxfmm-main"
for path in (SRC_DIR, JAXFMM_DIR):
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)

from config import DEFAULT_YUKAWA_CONFIG
from opendust_jax import (
    CylinderDomain,
    build_yukawa_tree,
    compute_force_metrics,
    direct_yukawa_forces,
    sample_uniform_cylinder,
    yukawa_fmm_forces,
)
from opendust_jax.plotting import plot_force_scatter


def _block_tree(tree: dict) -> None:
    for key in ("boxcenters", "eval_boxcenters", "boxlens", "mpl_cnct", "dir_cnct"):
        tree[key].block_until_ready()


def _evaluate_candidate(
    config,
    positions,
    charges,
    kappa: float,
    p: int,
    theta: float,
    n_max: int,
) -> tuple[dict[str, float], jnp.ndarray, jnp.ndarray]:
    t0 = time.perf_counter()
    tree = build_yukawa_tree(
        positions,
        n_max=n_max,
        theta=theta,
        p=p,
    )
    _block_tree(tree)
    tree_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    forces_fmm = yukawa_fmm_forces(
        positions,
        charges,
        kappa=kappa,
        eps0=config.eps0,
        tree=tree,
    )
    forces_fmm.block_until_ready()
    fmm_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    forces_direct = direct_yukawa_forces(
        positions,
        charges,
        kappa=kappa,
        eps0=config.eps0,
        batch_size=config.direct_batch_size,
    )
    forces_direct.block_until_ready()
    direct_time = time.perf_counter() - t0

    metrics = compute_force_metrics(forces_direct, forces_fmm)
    metrics_dict = metrics.to_dict()
    metrics_dict.update(
        {
            "p": p,
            "theta": theta,
            "n_max": n_max,
            "tree_build_s": tree_time,
            "yukawa_fmm_force_s": fmm_time,
            "direct_reference_s": direct_time,
            "acceptance_relative_l2": config.acceptance_relative_l2,
            "accepted": metrics.relative_l2 < config.acceptance_relative_l2,
        }
    )
    return metrics_dict, forces_direct, forces_fmm


def _run_case(config, positions, charges, debye_radius_m: float) -> dict[str, float]:
    kappa = 1.0 / debye_radius_m
    case_name = f"rD_{debye_radius_m:.6e}_m".replace("+", "")
    output_dir = Path(config.output_dir) / case_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nYukawa validation: r_D={debye_radius_m:.6e} m, kappa={kappa:.6e} 1/m")

    attempts = []
    best_metrics = None
    best_forces_direct = None
    best_forces_fmm = None
    for p, theta, n_max in config.accuracy_candidates:
        print(f"  candidate: p={p}, theta={theta}, n_max={n_max}")
        metrics_dict, forces_direct, forces_fmm = _evaluate_candidate(
            config,
            positions,
            charges,
            kappa,
            p,
            theta,
            n_max,
        )
        attempts.append(metrics_dict)
        print(
            "  rel L2={:.3e}, accepted={}".format(
                metrics_dict["relative_l2"],
                metrics_dict["accepted"],
            )
        )
        if best_metrics is None or metrics_dict["relative_l2"] < best_metrics["relative_l2"]:
            best_metrics = metrics_dict
            best_forces_direct = forces_direct
            best_forces_fmm = forces_fmm
        if metrics_dict["accepted"]:
            break

    assert best_metrics is not None
    assert best_forces_direct is not None
    assert best_forces_fmm is not None
    metrics_dict = dict(best_metrics)
    metrics_dict.update(
        {
            "n_particles": config.n_particles,
            "radius_m": config.radius_m,
            "height_m": config.height_m,
            "charge_c": config.charge_c,
            "seed": config.seed,
            "debye_radius_m": debye_radius_m,
            "kappa_1_per_m": kappa,
            "attempts": attempts,
        }
    )

    np.save(output_dir / "forces_direct.npy", np.asarray(best_forces_direct))
    np.save(output_dir / "forces_yukawa_fmm.npy", np.asarray(best_forces_fmm))
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics_dict, metrics_file, indent=2, sort_keys=True)

    title = (
        f"Yukawa FMM validation: N={config.n_particles}, p={metrics_dict['p']}, "
        f"theta={metrics_dict['theta']}, r_D={debye_radius_m:.2e} m, "
        f"rel L2={metrics_dict['relative_l2']:.3e}"
    )
    plot_force_scatter(
        best_forces_direct,
        best_forces_fmm,
        output_dir / "force_scatter.png",
        title=title,
    )
    print(json.dumps(metrics_dict, indent=2, sort_keys=True))
    return metrics_dict


def main() -> None:
    config = DEFAULT_YUKAWA_CONFIG
    domain = CylinderDomain(R=config.radius_m, H=config.height_m)
    positions = sample_uniform_cylinder(domain, config.n_particles, seed=config.seed)
    charges = jnp.full((config.n_particles,), config.charge_c)

    results = []
    for factor in config.debye_radius_factors:
        results.append(_run_case(config, positions, charges, config.radius_m * factor))

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as summary_file:
        json.dump(results, summary_file, indent=2, sort_keys=True)

    failed = [item for item in results if not item["accepted"]]
    if failed:
        raise SystemExit(f"Yukawa validation failed for {len(failed)} case(s).")


if __name__ == "__main__":
    main()
