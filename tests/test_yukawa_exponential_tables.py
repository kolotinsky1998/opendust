import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from opendust_jax.yukawa_exponential_tables import (
    apply_exponential_m2l_table,
    build_yukawa_m2l_exponential_table,
    exponential_m2l_operator_from_table,
    validate_yukawa_m2l_exponential_table,
)
from opendust_jax.yukawa_fmm import _spherical_m2l_coefficients_for_pair_rotated


def _analytic_m2l_operator(displacement, kappa_h, order):
    n_coeff = (order + 1) ** 2
    source_center = jnp.zeros((3,), dtype=jnp.float32)
    target_center = jnp.asarray(displacement, dtype=jnp.float32)
    columns = []
    for idx in range(n_coeff):
        basis = jnp.zeros((n_coeff,), dtype=jnp.complex64).at[idx].set(1.0 + 0.0j)
        columns.append(
            _spherical_m2l_coefficients_for_pair_rotated(
                target_center,
                source_center,
                basis,
                float(kappa_h),
                order,
            )
        )
    return jnp.stack(columns, axis=1)


@pytest.mark.parametrize("kappa_h", [0.25, 0.5, 1.0, 2.0])
def test_yukawa_exponential_table_shapes_for_p4(kappa_h):
    table = build_yukawa_m2l_exponential_table(
        order=4,
        kappa_h=kappa_h,
        direction="+z",
        accuracy="1e-3",
    )

    n_coeff = (table.order + 1) ** 2

    assert table.m2e.shape[1] == n_coeff
    assert table.e2l.shape[0] == n_coeff
    assert table.m2e.shape[0] == table.e2l.shape[1]
    assert table.nodes.shape[0] == table.weights.shape[0]
    assert table.metadata["s_exp"] == table.m2e.shape[0]


def test_yukawa_exponential_table_cache_returns_equivalent_table():
    first = build_yukawa_m2l_exponential_table(4, 0.5, "+z", accuracy="1e-3")
    second = build_yukawa_m2l_exponential_table(4, 0.5, "+z", accuracy="1e-3")

    assert first is second


def test_yukawa_exponential_table_matches_analytic_m2l_at_nodes():
    order = 4
    kappa_h = 1.0
    table = build_yukawa_m2l_exponential_table(order, kappa_h, "+z", accuracy="1e-3")

    for displacement in np.asarray(table.nodes[:5]):
        approx = exponential_m2l_operator_from_table(table, displacement)
        reference = _analytic_m2l_operator(displacement, kappa_h, order)
        rel = np.linalg.norm(np.asarray(approx - reference)) / max(
            np.linalg.norm(np.asarray(reference)),
            1.0e-30,
        )
        assert rel < 1.0e-3


def test_yukawa_exponential_apply_matches_operator_product_at_node():
    order = 4
    kappa_h = 1.0
    table = build_yukawa_m2l_exponential_table(order, kappa_h, "+z", accuracy="1e-3")
    displacement = table.nodes[0]
    n_coeff = (order + 1) ** 2
    source_moments = jnp.asarray(
        [complex(np.sin(0.17 * idx), np.cos(0.11 * idx)) / (1.0 + idx) for idx in range(n_coeff)],
        dtype=jnp.complex64,
    )

    approx = apply_exponential_m2l_table(table, source_moments, displacement)
    reference_op = _analytic_m2l_operator(displacement, kappa_h, order)
    reference = reference_op @ source_moments

    np.testing.assert_allclose(np.asarray(approx), np.asarray(reference), rtol=1e-3, atol=1e-6)


@pytest.mark.parametrize("direction", ["+z", "-z", "+x", "-x", "+y", "-y"])
def test_yukawa_exponential_table_validates_all_directions(direction):
    table = build_yukawa_m2l_exponential_table(4, 0.5, direction, accuracy="1e-3")

    validation = validate_yukawa_m2l_exponential_table(table)

    assert validation["max_relative_frobenius_error"] < 1.0e-3


def test_yukawa_exponential_table_rejects_unknown_accuracy():
    with pytest.raises(ValueError, match="accuracy"):
        build_yukawa_m2l_exponential_table(4, 0.5, "+z", accuracy="fast")


def test_yukawa_exponential_table_rejects_too_small_s_exp_max():
    with pytest.raises(RuntimeError, match="S_exp"):
        build_yukawa_m2l_exponential_table(
            4,
            0.5,
            "+z",
            accuracy="1e-3",
            s_exp_max=10,
        )
