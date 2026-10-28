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

from opendust_jax.pair_forces import (
    softened_yukawa_forces,
    softened_yukawa_forces_2d,
)


def numpy_reference(
    positions: np.ndarray,
    prefactor: float,
    screening_length: float,
    softening: float,
) -> np.ndarray:
    """Independent float64 reference implementation."""

    positions = np.asarray(positions, dtype=np.float64)
    forces = np.zeros_like(positions)
    inverse_screening_length = 1.0 / screening_length
    for target in range(len(positions)):
        for source in range(len(positions)):
            if target == source:
                continue
            displacement = positions[target] - positions[source]
            distance = np.linalg.norm(displacement)
            denominator = distance + softening
            coefficient = (
                prefactor
                * np.exp(-distance * inverse_screening_length)
                * (
                    inverse_screening_length / denominator
                    + 1.0 / denominator**2
                )
                / distance
            )
            forces[target] += coefficient * displacement
    return forces


def test_single_particle_has_zero_force_and_no_nan():
    forces = softened_yukawa_forces(
        jnp.zeros((1, 3)), 1.0, 1.0, 1.0e-3, block_size=8
    )
    actual = np.asarray(forces)
    np.testing.assert_array_equal(actual, np.zeros((1, 3), dtype=np.float32))
    assert np.all(np.isfinite(actual))


def test_two_particles_obey_newtons_third_law():
    positions = jnp.array([[0.0, 0.0, 0.0], [0.4, -0.2, 0.1]])
    forces = np.asarray(
        softened_yukawa_forces(positions, 2.0, 0.8, 1.0e-3, block_size=2)
    )
    np.testing.assert_allclose(forces[0], -forces[1], rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("block_size", [1, 2, 4, 8])
def test_matches_numpy_for_non_multiple_particle_count(block_size):
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.7, 0.1, -0.2],
            [-0.3, 1.1, 0.4],
            [0.2, -0.6, 0.8],
            [1.3, 0.2, -0.4],
        ],
        dtype=np.float32,
    )
    expected = numpy_reference(positions, 2.5, 0.8, 0.03)
    actual = softened_yukawa_forces(
        positions, 2.5, 0.8, 0.03, block_size=block_size
    )
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=5e-6, atol=2e-6)


def test_total_internal_force_is_nearly_zero():
    positions = jax.random.normal(jax.random.PRNGKey(4), (31, 3))
    forces = softened_yukawa_forces(
        positions, 1.0, 1.3, 1.0e-3, block_size=8
    )
    total = np.asarray(jnp.sum(forces, axis=0))
    scale = float(jnp.sum(jnp.linalg.norm(forces, axis=1)))
    assert np.linalg.norm(total) <= 2e-6 * scale


@pytest.mark.parametrize("target_block_size", [1, 3, 8])
@pytest.mark.parametrize("source_block_size", [1, 2, 7])
def test_2d_tiling_matches_numpy_and_1d_backend(
    target_block_size, source_block_size
):
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.7, 0.1, -0.2],
            [-0.3, 1.1, 0.4],
            [0.2, -0.6, 0.8],
            [1.3, 0.2, -0.4],
        ],
        dtype=np.float32,
    )
    expected = numpy_reference(positions, 2.5, 0.8, 0.03)
    baseline = softened_yukawa_forces(
        positions, 2.5, 0.8, 0.03, block_size=4
    )
    actual = softened_yukawa_forces_2d(
        positions,
        2.5,
        0.8,
        0.03,
        target_block_size=target_block_size,
        source_block_size=source_block_size,
    )
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=5e-6, atol=2e-6)
    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(baseline), rtol=5e-6, atol=2e-6
    )


def test_2d_tiling_single_particle_is_finite_and_zero():
    actual = softened_yukawa_forces_2d(
        jnp.zeros((1, 3)),
        1.0,
        1.0,
        1.0e-3,
        target_block_size=8,
        source_block_size=7,
    )
    np.testing.assert_array_equal(
        np.asarray(actual), np.zeros((1, 3), dtype=np.float32)
    )
    assert np.all(np.isfinite(np.asarray(actual)))


def test_rejects_invalid_shape_and_block_size():
    with pytest.raises(ValueError, match="shape"):
        softened_yukawa_forces(jnp.zeros((3, 2)), 1.0, 1.0, 0.0)
    with pytest.raises(ValueError, match="block_size"):
        softened_yukawa_forces(jnp.zeros((3, 3)), 1.0, 1.0, 0.0, block_size=0)
    with pytest.raises(ValueError, match="screening_length"):
        softened_yukawa_forces(jnp.zeros((3, 3)), 1.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="softening"):
        softened_yukawa_forces(jnp.zeros((3, 3)), 1.0, 1.0, -1.0)
    with pytest.raises(ValueError, match="target_block_size"):
        softened_yukawa_forces_2d(
            jnp.zeros((3, 3)),
            1.0,
            1.0,
            0.0,
            target_block_size=0,
        )
    with pytest.raises(ValueError, match="source_block_size"):
        softened_yukawa_forces_2d(
            jnp.zeros((3, 3)),
            1.0,
            1.0,
            0.0,
            source_block_size=0,
        )
