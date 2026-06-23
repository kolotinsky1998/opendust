import sys
from pathlib import Path

import jax.numpy as jnp
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
JAXFMM_DIR = REPO_ROOT / "jaxfmm-main"
for path in (SRC_DIR, JAXFMM_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

pytest.importorskip("jaxfmm")

from opendust_jax import (
    CylinderDomain,
    build_fmm_tree,
    build_yukawa_tree,
    compute_force_metrics,
    direct_coulomb_forces,
    direct_yukawa_forces,
    fmm_coulomb_forces,
    sample_uniform_cylinder,
    yukawa_fmm_forces,
)


def test_fmm_coulomb_forces_match_direct_reference_smoke():
    domain = CylinderDomain(R=5.0e-4, H=1.0e-3)
    positions = sample_uniform_cylinder(domain, 512, seed=11)
    charges = jnp.full((512,), 1.60217662e-19)

    tree = build_fmm_tree(positions, n_max=64, theta=0.77, p=4)
    direct = direct_coulomb_forces(positions, charges, batch_size=128)
    fmm = fmm_coulomb_forces(positions, charges, tree=tree)
    metrics = compute_force_metrics(direct, fmm)

    assert metrics.relative_l2 < 2.0e-2


def test_yukawa_fmm_forces_match_direct_reference_smoke():
    domain = CylinderDomain(R=5.0e-4, H=1.0e-3)
    positions = sample_uniform_cylinder(domain, 256, seed=12)
    charges = jnp.full((256,), 1.60217662e-19)
    kappa = 1.0 / domain.R

    tree = build_yukawa_tree(positions, n_max=64, theta=0.77, p=4)
    direct = direct_yukawa_forces(positions, charges, kappa=kappa, batch_size=64)
    fmm = yukawa_fmm_forces(positions, charges, kappa=kappa, tree=tree, backend="chebyshev")
    metrics = compute_force_metrics(direct, fmm)

    assert metrics.relative_l2 < 5.0e-2


def test_yukawa_spherical_local_backend_smoke():
    domain = CylinderDomain(R=5.0e-4, H=1.0e-3)
    positions = sample_uniform_cylinder(domain, 256, seed=13)
    charges = jnp.full((256,), 1.60217662e-19)
    kappa = 1.0 / domain.R

    tree = build_yukawa_tree(positions, n_max=64, theta=0.45, p=4)
    direct = direct_yukawa_forces(positions, charges, kappa=kappa, batch_size=64)
    fmm = yukawa_fmm_forces(
        positions,
        charges,
        kappa=kappa,
        tree=tree,
        backend="spherical_local",
        local_order=1,
    )
    metrics = compute_force_metrics(direct, fmm)

    assert metrics.relative_l2 < 1.0e-1


def test_yukawa_spherical_m2l_backend_smoke():
    domain = CylinderDomain(R=5.0e-4, H=1.0e-3)
    positions = sample_uniform_cylinder(domain, 256, seed=14)
    charges = jnp.full((256,), 1.60217662e-19)
    kappa = 1.0 / domain.R

    tree = build_yukawa_tree(positions, n_max=64, theta=0.45, p=4)
    direct = direct_yukawa_forces(positions, charges, kappa=kappa, batch_size=64)
    fmm = yukawa_fmm_forces(
        positions,
        charges,
        kappa=kappa,
        tree=tree,
        backend="spherical_m2l",
    )
    metrics = compute_force_metrics(direct, fmm)

    assert metrics.relative_l2 < 1.0e-1


def test_yukawa_spherical_multilevel_backend_smoke():
    domain = CylinderDomain(R=5.0e-4, H=1.0e-3)
    positions = sample_uniform_cylinder(domain, 256, seed=15)
    charges = jnp.full((256,), 1.60217662e-19)
    kappa = 1.0 / domain.R

    tree = build_yukawa_tree(positions, n_max=64, theta=0.45, p=4)
    direct = direct_yukawa_forces(positions, charges, kappa=kappa, batch_size=64)
    fmm = yukawa_fmm_forces(
        positions,
        charges,
        kappa=kappa,
        tree=tree,
        backend="spherical_multilevel",
    )
    metrics = compute_force_metrics(direct, fmm)

    assert metrics.relative_l2 < 1.0e-1
