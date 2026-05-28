import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from opendust_jax.direct import direct_coulomb_field, direct_coulomb_forces
from opendust_jax.geometry import CylinderDomain, points_inside_cylinder, sample_uniform_cylinder
from opendust_jax.validation import compute_force_metrics


def test_sample_uniform_cylinder_points_are_inside():
    domain = CylinderDomain(R=2.0, H=5.0)
    points = sample_uniform_cylinder(domain, 1024, seed=7)

    assert bool(jnp.all(points_inside_cylinder(points, domain)))


def test_single_particle_has_zero_self_force():
    positions = jnp.array([[0.0, 0.0, 0.0]])
    charges = jnp.array([1.0])

    forces = direct_coulomb_forces(positions, charges, eps0=1.0)

    np.testing.assert_allclose(np.asarray(forces), np.zeros((1, 3)), atol=0.0)


def test_two_equal_charges_have_equal_and_opposite_forces():
    positions = jnp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    charges = jnp.array([3.0, 3.0])

    forces = direct_coulomb_forces(positions, charges, eps0=1.0)

    np.testing.assert_allclose(np.asarray(forces[0]), -np.asarray(forces[1]), rtol=1e-6)
    assert forces[0, 0] < 0.0
    assert forces[1, 0] > 0.0


def test_direct_solver_matches_numpy_reference():
    eps0 = 2.5
    positions = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ]
    )
    charges = jnp.array([1.0, 2.0, 4.0])

    field = direct_coulomb_field(positions, charges, eps0=eps0)

    positions_np = np.asarray(positions)
    charges_np = np.asarray(charges)
    expected = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            diff = positions_np[i] - positions_np[j]
            r = np.linalg.norm(diff)
            expected[i] += charges_np[j] * diff / r**3
    expected /= 4.0 * np.pi * eps0

    np.testing.assert_allclose(np.asarray(field), expected, rtol=1e-6, atol=1e-7)


def test_compute_force_metrics_zero_error():
    forces = jnp.array([[1.0, 2.0, 3.0], [-2.0, 5.0, 7.0]])

    metrics = compute_force_metrics(forces, forces)

    assert metrics.relative_l2 == 0.0
    assert metrics.max_abs_error == 0.0
