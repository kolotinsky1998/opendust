import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from opendust_jax.direct import (
    direct_coulomb_field,
    direct_coulomb_forces,
    direct_yukawa_field,
    direct_yukawa_forces,
)
from opendust_jax.geometry import CylinderDomain, points_inside_cylinder, sample_uniform_cylinder
from opendust_jax.validation import compute_force_metrics
from opendust_jax.yukawa_basis import modified_spherical_bessel_i, modified_spherical_bessel_k
from opendust_jax.yukawa_fmm import (
    _compute_spherical_moments,
    _sphere_projection_quadrature,
    _spherical_m2m_coefficients_closed,
    _spherical_m2m_coefficients_projected,
    _spherical_l2l_coefficients_closed,
    _spherical_l2l_coefficients_projected,
    _spherical_regular_couplings,
    _spherical_m2l_coefficients_for_pair_projected,
    _spherical_yukawa_local_field_from_coeffs,
    _spherical_yukawa_field_from_moments,
    _spherical_yukawa_potential_from_moments,
)


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


def test_single_particle_has_zero_yukawa_self_force():
    positions = jnp.array([[0.0, 0.0, 0.0]])
    charges = jnp.array([1.0])

    forces = direct_yukawa_forces(positions, charges, kappa=2.0, eps0=1.0)

    np.testing.assert_allclose(np.asarray(forces), np.zeros((1, 3)), atol=0.0)


def test_two_equal_charges_have_equal_and_opposite_yukawa_forces():
    positions = jnp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    charges = jnp.array([3.0, 3.0])

    forces = direct_yukawa_forces(positions, charges, kappa=0.5, eps0=1.0)

    np.testing.assert_allclose(np.asarray(forces[0]), -np.asarray(forces[1]), rtol=1e-6)
    assert forces[0, 0] < 0.0
    assert forces[1, 0] > 0.0


def test_yukawa_approaches_coulomb_for_small_kappa():
    positions = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ]
    )
    charges = jnp.array([1.0, 2.0, 4.0])

    coulomb = direct_coulomb_field(positions, charges, eps0=1.0)
    yukawa = direct_yukawa_field(positions, charges, kappa=1e-8, eps0=1.0)

    np.testing.assert_allclose(np.asarray(yukawa), np.asarray(coulomb), rtol=1e-5, atol=1e-6)


def test_yukawa_radial_basis_is_finite_near_zero():
    x = jnp.array([0.0, 1e-10, 1e-4, 1.0])

    regular = modified_spherical_bessel_i(x, 4)
    singular = modified_spherical_bessel_k(x + 1e-6, 4)

    assert bool(jnp.all(jnp.isfinite(regular)))
    assert bool(jnp.all(jnp.isfinite(singular)))


def test_yukawa_radial_i_small_x_leading_terms():
    x = jnp.array([1e-5])
    vals = modified_spherical_bessel_i(x, 4)[0]

    np.testing.assert_allclose(float(vals[0]), 1.0, rtol=1e-6)
    np.testing.assert_allclose(float(vals[1]), float(x[0] / 3.0), rtol=1e-5)
    np.testing.assert_allclose(float(vals[2]), float(x[0] ** 2 / 15.0), rtol=1e-5)


def test_spherical_yukawa_monopole_matches_direct_kernel():
    charge = 2.5
    kappa = 3.0
    center = jnp.array([[0.0, 0.0, 0.0]])
    source = jnp.array([[[0.0, 0.0, 0.0]]])
    charges = jnp.array([[charge]])
    target = jnp.array([0.7, 0.2, 0.4])

    moments = _compute_spherical_moments(source, charges, center, kappa, 0)[0]
    potential = _spherical_yukawa_potential_from_moments(
        target,
        center[0],
        moments,
        kappa,
        0,
    )
    r = jnp.linalg.norm(target)
    expected = charge * jnp.exp(-kappa * r) / r

    np.testing.assert_allclose(float(potential), float(expected), rtol=1e-5)


def test_spherical_yukawa_monopole_field_matches_direct_kernel():
    charge = 2.5
    kappa = 3.0
    center = jnp.array([[0.0, 0.0, 0.0]])
    source = jnp.array([[[0.0, 0.0, 0.0]]])
    charges = jnp.array([[charge]])
    target = jnp.array([0.7, 0.2, 0.4])

    moments = _compute_spherical_moments(source, charges, center, kappa, 0)[0]
    field = _spherical_yukawa_field_from_moments(
        target,
        center[0],
        moments,
        kappa,
        0,
    )
    r = jnp.linalg.norm(target)
    expected = charge * jnp.exp(-kappa * r) * (1.0 / r**3 + kappa / r**2) * target

    np.testing.assert_allclose(np.asarray(field), np.asarray(expected), rtol=2e-5)


def test_projected_yukawa_m2l_reproduces_monopole_m2p_field():
    charge = 2.5
    kappa = 3.0
    order = 4
    source_center = jnp.array([0.0, 0.0, 0.0])
    target_center = jnp.array([1.0, 0.3, -0.2])
    source = jnp.array([[[0.0, 0.0, 0.0]]])
    charges = jnp.array([[charge]])

    moments = _compute_spherical_moments(
        source,
        charges,
        source_center[None, :],
        kappa,
        order,
    )[0]
    quad_dirs, quad_weights = _sphere_projection_quadrature(order)
    local_coeffs = _spherical_m2l_coefficients_for_pair_projected(
        target_center,
        source_center,
        moments,
        quad_dirs,
        quad_weights,
        kappa,
        order,
    )

    offsets = jnp.array(
        [
            [0.01, 0.00, 0.00],
            [0.00, -0.02, 0.01],
            [-0.01, 0.01, -0.015],
        ]
    )
    points = target_center + offsets
    m2p_field = jnp.stack(
        [
            _spherical_yukawa_field_from_moments(
                point,
                source_center,
                moments,
                kappa,
                order,
            )
            for point in points
        ]
    )
    local_field = jnp.stack(
        [
            _spherical_yukawa_local_field_from_coeffs(
                point,
                target_center,
                local_coeffs,
                kappa,
                order,
            )
            for point in points
        ]
    )

    np.testing.assert_allclose(np.asarray(local_field), np.asarray(m2p_field), rtol=1e-3, atol=1e-6)


@pytest.mark.xfail(reason="Closed Yukawa M2M translation is not derived correctly yet.")
def test_closed_yukawa_m2m_matches_projection_oracle():
    kappa = 3.0
    order = 4
    child_center = jnp.array([0.2, -0.1, 0.05])
    parent_center = jnp.array([0.0, 0.0, 0.0])
    source = jnp.array(
        [
            [
                [0.21, -0.09, 0.04],
                [0.19, -0.12, 0.055],
                [0.205, -0.08, 0.07],
            ]
        ]
    )
    charges = jnp.array([[2.0, -0.5, 1.2]])

    child_moments = _compute_spherical_moments(
        source,
        charges,
        child_center[None, :],
        kappa,
        order,
    )[0]
    reg_local_indices, reg_source_indices, reg_big_basis_indices, reg_coupling_coeffs = (
        _spherical_regular_couplings(order)
    )
    closed = _spherical_m2m_coefficients_closed(
        parent_center,
        child_center,
        child_moments,
        reg_local_indices,
        reg_source_indices,
        reg_big_basis_indices,
        reg_coupling_coeffs,
        kappa,
        order,
    )

    quad_dirs, quad_weights = _sphere_projection_quadrature(order)
    projected = _spherical_m2m_coefficients_projected(
        parent_center,
        child_center,
        child_moments,
        quad_dirs,
        quad_weights,
        kappa,
        order,
    )

    np.testing.assert_allclose(np.asarray(closed), np.asarray(projected), rtol=1e-3, atol=1e-8)


def test_closed_yukawa_l2l_matches_projection_oracle():
    kappa = 3.0
    order = 4
    parent_center = jnp.array([0.0, 0.0, 0.0])
    child_center = jnp.array([0.12, -0.08, 0.04])
    source_center = jnp.array([-1.0, 0.4, 0.2])
    source = jnp.array(
        [
            [
                [-1.01, 0.39, 0.21],
                [-0.98, 0.42, 0.18],
                [-1.03, 0.38, 0.19],
            ]
        ]
    )
    charges = jnp.array([[2.0, -0.5, 1.2]])

    source_moments = _compute_spherical_moments(
        source,
        charges,
        source_center[None, :],
        kappa,
        order,
    )[0]
    quad_dirs, quad_weights = _sphere_projection_quadrature(order)
    parent_coeffs = _spherical_m2l_coefficients_for_pair_projected(
        parent_center,
        source_center,
        source_moments,
        quad_dirs,
        quad_weights,
        kappa,
        order,
    )

    reg_local_indices, reg_source_indices, reg_big_basis_indices, reg_coupling_coeffs = (
        _spherical_regular_couplings(order)
    )
    closed = _spherical_l2l_coefficients_closed(
        child_center,
        parent_center,
        parent_coeffs,
        reg_local_indices,
        reg_source_indices,
        reg_big_basis_indices,
        reg_coupling_coeffs,
        kappa,
        order,
    )

    projected = _spherical_l2l_coefficients_projected(
        child_center,
        parent_center,
        parent_coeffs,
        quad_dirs,
        quad_weights,
        kappa,
        order,
    )

    np.testing.assert_allclose(np.asarray(closed), np.asarray(projected), rtol=1e-3, atol=1e-8)
