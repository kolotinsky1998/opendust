"""Run Coulomb direct-vs-FMM validation in a cylindrical source domain."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import jax.numpy as jnp

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
SRC_DIR = REPO_ROOT / "src"
JAXFMM_DIR = REPO_ROOT / "jaxfmm-main"
for path in (SRC_DIR, JAXFMM_DIR):
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)

from config import DEFAULT_CONFIG
from opendust_jax import (
    CylinderDomain,
    build_fmm_tree,
    compute_force_metrics,
    direct_coulomb_forces,
    fmm_coulomb_forces,
    sample_uniform_cylinder,
)
from opendust_jax.plotting import plot_force_scatter


def main() -> None:
    config = DEFAULT_CONFIG
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    domain = CylinderDomain(R=config.radius_m, H=config.height_m)
    positions = sample_uniform_cylinder(domain, config.n_particles, seed=config.seed)
    charges = jnp.full((config.n_particles,), config.charge_c)

    print("Building FMM hierarchy...")
    tree = build_fmm_tree(
        positions,
        n_max=config.n_max,
        theta=config.theta,
        p=config.p,
    )

    print("Computing FMM forces...")
    forces_fmm = fmm_coulomb_forces(
        positions,
        charges,
        eps0=config.eps0,
        tree=tree,
    )

    print("Computing direct reference forces...")
    forces_direct = direct_coulomb_forces(
        positions,
        charges,
        eps0=config.eps0,
        batch_size=config.direct_batch_size,
    )

    metrics = compute_force_metrics(forces_direct, forces_fmm)
    metrics_dict = metrics.to_dict()
    metrics_dict.update(
        {
            "n_particles": config.n_particles,
            "radius_m": config.radius_m,
            "height_m": config.height_m,
            "charge_c": config.charge_c,
            "seed": config.seed,
            "p": config.p,
            "theta": config.theta,
            "n_max": config.n_max,
            "acceptance_relative_l2": config.acceptance_relative_l2,
            "accepted": metrics.relative_l2 < config.acceptance_relative_l2,
        }
    )

    np.save(output_dir / "forces_direct.npy", np.asarray(forces_direct))
    np.save(output_dir / "forces_fmm.npy", np.asarray(forces_fmm))
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics_dict, metrics_file, indent=2, sort_keys=True)

    title = (
        f"Coulomb FMM validation: N={config.n_particles}, "
        f"p={config.p}, theta={config.theta}, "
        f"rel L2={metrics.relative_l2:.3e}"
    )
    plot_force_scatter(
        forces_direct,
        forces_fmm,
        output_dir / "force_scatter.png",
        title=title,
    )

    print(json.dumps(metrics_dict, indent=2, sort_keys=True))
    if metrics.relative_l2 >= config.acceptance_relative_l2:
        raise SystemExit(
            "Coulomb FMM validation failed: "
            f"relative_l2={metrics.relative_l2:.3e} >= {config.acceptance_relative_l2:.3e}"
        )


if __name__ == "__main__":
    main()
