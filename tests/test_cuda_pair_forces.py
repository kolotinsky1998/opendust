import sys
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from opendust_jax.cuda_pair_forces import (
    cuda_pair_forces_available,
    dust_orbit_forces_cuda,
    external_forces_cuda,
    ion_dust_forces_cuda,
    softened_yukawa_forces_cuda,
    softened_yukawa_forces_cuda_with_stats,
)
from opendust_jax.external_forces import (
    dust_orbit_forces_jax,
    external_forces_jax,
    ion_dust_forces_jax,
)
from opendust_jax.physics import background_force_N


def numpy_reference(positions, prefactor, screening_length, softening):
    positions = np.asarray(positions, dtype=np.float64)
    result = np.zeros_like(positions)
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            displacement = positions[i] - positions[j]
            distance = np.linalg.norm(displacement)
            denominator = distance + softening
            coefficient = (
                prefactor
                * np.exp(-distance / screening_length)
                * (
                    1.0 / (screening_length * denominator)
                    + 1.0 / denominator**2
                )
                / distance
            )
            force = coefficient * displacement
            result[i] += force
            result[j] -= force
    return result


def test_cuda_frontend_rejects_invalid_shape_before_loading_library():
    with pytest.raises(ValueError, match="shape"):
        softened_yukawa_forces_cuda(jnp.zeros((4, 2)), 1.0, 1.0, 0.0)


def test_split_external_jax_components_match_fused_reference():
    rng = np.random.default_rng(90)
    positions = jnp.asarray(rng.normal(size=(13, 3)), jnp.float32)
    dust_positions = jnp.asarray([[0.2, -0.1, 0.3], [-0.4, 0.2, -0.5]], jnp.float32)
    dust_charges = jnp.asarray([-0.7, -1.1], jnp.float32)
    coefficients = jnp.asarray(rng.normal(scale=1e-4, size=15), jnp.float32)
    ion_charge, screening_length, electric_field = 1.3, 0.8, 0.25
    fused_force, fused_orbit = external_forces_jax(
        positions,
        dust_positions,
        dust_charges,
        coefficients,
        ion_charge,
        screening_length,
        electric_field,
    )
    fast = ion_dust_forces_jax(
        positions, dust_positions, dust_charges, ion_charge, electric_field
    )
    background = background_force_N(
        positions, ion_charge, coefficients, screening_length, ion_charge
    )
    orbit = dust_orbit_forces_jax(
        positions, dust_positions, dust_charges, ion_charge, screening_length
    )
    np.testing.assert_allclose(fast + background, fused_force, rtol=2e-6)
    np.testing.assert_allclose(orbit, fused_orbit, rtol=2e-6)


@pytest.mark.skipif(
    not cuda_pair_forces_available(),
    reason="CUDA FFI pair-force library is not built on a JAX GPU host",
)
def test_cuda_matches_numpy_for_non_multiple_particle_count_under_jit():
    rng = np.random.default_rng(81)
    positions = rng.normal(size=(37, 3)).astype(np.float32)
    expected = numpy_reference(positions, 1.7, 0.9, 1.0e-3)

    operation = jax.jit(softened_yukawa_forces_cuda)
    actual = np.asarray(operation(positions, 1.7, 0.9, 1.0e-3))

    np.testing.assert_allclose(actual, expected, rtol=5.0e-5, atol=2.0e-4)
    assert np.all(np.isfinite(actual))
    total = np.sum(actual.astype(np.float64), axis=0)
    scale = np.sum(np.linalg.norm(actual.astype(np.float64), axis=1))
    assert np.linalg.norm(total) <= 5.0e-6 * scale


@pytest.mark.skipif(
    not cuda_pair_forces_available(),
    reason="CUDA FFI pair-force library is not built on a JAX GPU host",
)
def test_cuda_single_particle_is_zero():
    actual = softened_yukawa_forces_cuda(
        jnp.zeros((1, 3)), 1.0, 1.0, 1.0e-3
    )
    np.testing.assert_array_equal(
        np.asarray(actual), np.zeros((1, 3), dtype=np.float32)
    )


@pytest.mark.skipif(
    not cuda_pair_forces_available(),
    reason="CUDA FFI pair-force library is not built on a JAX GPU host",
)
def test_cuda_coincident_distinct_particles_are_finite_and_counted():
    positions = jnp.zeros((2, 3), dtype=jnp.float32)
    forces, count = softened_yukawa_forces_cuda_with_stats(
        positions, 1.0, 1.0, 1.0e-3
    )
    np.testing.assert_array_equal(np.asarray(forces), np.zeros((2, 3), np.float32))
    assert int(count) == 1


@pytest.mark.skipif(
    not cuda_pair_forces_available(),
    reason="CUDA FFI force library is not built on a JAX GPU host",
)
def test_fused_external_cuda_matches_jax_reference():
    rng = np.random.default_rng(91)
    positions = jnp.asarray(rng.normal(size=(67, 3)), jnp.float32)
    dust_positions = jnp.asarray([[0.2, -0.1, 0.3], [-0.4, 0.2, -0.5]], jnp.float32)
    dust_charges = jnp.asarray([-0.7, -1.1], jnp.float32)
    coefficients = jnp.asarray(rng.normal(scale=1e-4, size=15), jnp.float32)
    parameters = (1.3, 0.8, 0.25)
    expected = external_forces_jax(
        positions, dust_positions, dust_charges, coefficients, *parameters
    )
    actual = jax.jit(external_forces_cuda)(
        positions, dust_positions, dust_charges, coefficients, *parameters
    )
    np.testing.assert_allclose(
        np.asarray(actual[0]), np.asarray(expected[0]), rtol=2e-5, atol=3e-4
    )
    np.testing.assert_allclose(
        np.asarray(actual[1]), np.asarray(expected[1]), rtol=5e-5, atol=3e-4
    )


@pytest.mark.skipif(
    not cuda_pair_forces_available(),
    reason="CUDA FFI force library is not built on a JAX GPU host",
)
def test_split_external_cuda_matches_jax_reference():
    rng = np.random.default_rng(92)
    positions = jnp.asarray(rng.normal(size=(67, 3)), jnp.float32)
    dust_positions = jnp.asarray([[0.2, -0.1, 0.3], [-0.4, 0.2, -0.5]], jnp.float32)
    dust_charges = jnp.asarray([-0.7, -1.1], jnp.float32)
    ion_charge, screening_length, electric_field = 1.3, 0.8, 0.25
    expected_fast = ion_dust_forces_jax(
        positions, dust_positions, dust_charges, ion_charge, electric_field
    )
    expected_orbit = dust_orbit_forces_jax(
        positions, dust_positions, dust_charges, ion_charge, screening_length
    )
    actual_fast = jax.jit(ion_dust_forces_cuda)(
        positions, dust_positions, dust_charges, ion_charge, electric_field
    )
    actual_orbit = jax.jit(dust_orbit_forces_cuda)(
        positions, dust_positions, dust_charges, ion_charge, screening_length
    )
    np.testing.assert_allclose(
        np.asarray(actual_fast), np.asarray(expected_fast), rtol=2e-5, atol=3e-4
    )
    np.testing.assert_allclose(
        np.asarray(actual_orbit), np.asarray(expected_orbit), rtol=5e-5, atol=3e-4
    )
