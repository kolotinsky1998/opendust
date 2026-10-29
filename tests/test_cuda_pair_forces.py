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
    softened_yukawa_forces_cuda,
)


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
