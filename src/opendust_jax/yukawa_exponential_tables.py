"""Operator tables for fast Yukawa M2L experiments.

This module intentionally stops at operator-level validation.  It builds a
deterministic table that approximates the verified spherical Yukawa M2L
operator for admissible box displacements.  The storage layout mirrors the
future exponential M2L path: source multipoles are expanded to table modes
(``M2E``), displacement-dependent diagonal weights are applied, and local
coefficients are reconstructed by ``E2L``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from opendust_jax.yukawa_fmm import _spherical_m2l_coefficients_for_pair_rotated


_VALID_DIRECTIONS = ("+z", "-z", "+x", "-x", "+y", "-y")
_ACCURACY_TARGETS = {"1e-3": 1.0e-3, "1e-6": 1.0e-6}


@dataclass(frozen=True)
class YukawaExponentialM2LTable:
    order: int
    kappa_h: float
    direction: str
    accuracy: str
    nodes: jax.Array
    weights: jax.Array
    m2e: jax.Array
    e2l: jax.Array
    metadata: dict[str, Any]


def _validate_inputs(order: int, kappa_h: float, direction: str, accuracy: str) -> None:
    if not 2 <= int(order) <= 8:
        raise ValueError("order must be in the supported range 2..8.")
    if float(kappa_h) <= 0.0:
        raise ValueError("kappa_h must be positive.")
    if direction not in _VALID_DIRECTIONS:
        raise ValueError(f"direction must be one of {_VALID_DIRECTIONS}.")
    if accuracy not in _ACCURACY_TARGETS:
        raise ValueError(f"accuracy must be one of {tuple(_ACCURACY_TARGETS)}.")


def _dtype_name(dtype: Any) -> str:
    return str(jnp.dtype(dtype))


def _dtype_from_name(name: str) -> jnp.dtype:
    return jnp.dtype(name)


def _axis_for_direction(direction: str) -> tuple[int, float]:
    axis_name = direction[1]
    axis = {"x": 0, "y": 1, "z": 2}[axis_name]
    sign = 1.0 if direction[0] == "+" else -1.0
    return axis, sign


def _direction_grid(direction: str, accuracy: str) -> tuple[np.ndarray, np.ndarray, int, float, tuple[int, int]]:
    axis, sign = _axis_for_direction(direction)
    main_values = (2.0, 2.5, 3.0, 4.0)
    transverse_values = (-0.75, 0.0, 0.75)
    if accuracy == "1e-6":
        main_values = (2.0, 2.25, 2.5, 3.0, 3.5, 4.0)
        transverse_values = (-1.0, -0.5, 0.0, 0.5, 1.0)
    other_axes = tuple(idx for idx in range(3) if idx != axis)
    return (
        np.asarray(main_values, dtype=np.float64),
        np.asarray(transverse_values, dtype=np.float64),
        axis,
        sign,
        other_axes,
    )


def _direction_nodes(direction: str, accuracy: str) -> np.ndarray:
    """Build deterministic admissible displacement nodes in box-size units."""

    main_values, transverse_values, axis, sign, other_axes = _direction_grid(direction, accuracy)
    nodes = []
    for main in main_values:
        for a in transverse_values:
            for b in transverse_values:
                vec = np.zeros(3, dtype=np.float64)
                vec[axis] = sign * main
                vec[other_axes[0]] = a
                vec[other_axes[1]] = b
                if abs(vec[axis]) >= max(abs(vec[other_axes[0]]), abs(vec[other_axes[1]])):
                    nodes.append(vec)
    return np.asarray(nodes, dtype=np.float64)


def _direction_off_node_displacements(direction: str, accuracy: str) -> np.ndarray:
    """Build holdout displacements between table nodes."""

    main_values, transverse_values, axis, sign, other_axes = _direction_grid(direction, accuracy)
    main_mid = 0.5 * (main_values[:-1] + main_values[1:])
    transverse_mid = 0.5 * (transverse_values[:-1] + transverse_values[1:])
    nodes = []
    for main in main_mid:
        for a in transverse_mid:
            for b in transverse_mid:
                vec = np.zeros(3, dtype=np.float64)
                vec[axis] = sign * main
                vec[other_axes[0]] = a
                vec[other_axes[1]] = b
                nodes.append(vec)
    return np.asarray(nodes, dtype=np.float64)


def _m2l_operator_for_displacement(displacement: np.ndarray, kappa_h: float, order: int) -> np.ndarray:
    n_coeff = (int(order) + 1) ** 2
    source_center = jnp.zeros((3,), dtype=jnp.float32)
    target_center = jnp.asarray(displacement, dtype=jnp.float32)
    basis = jnp.eye(n_coeff, dtype=jnp.complex64)
    operator = jax.vmap(
        lambda moments: _spherical_m2l_coefficients_for_pair_rotated(
            target_center,
            source_center,
            moments,
            float(kappa_h),
            int(order),
        )
    )(basis)
    return np.asarray(operator).T


def _build_raw_table(
    order: int,
    kappa_h: float,
    direction: str,
    accuracy: str,
    dtype_name: str,
    s_exp_max: int | None,
) -> YukawaExponentialM2LTable:
    _validate_inputs(order, kappa_h, direction, accuracy)
    dtype = _dtype_from_name(dtype_name)
    n_coeff = (int(order) + 1) ** 2
    nodes_np = _direction_nodes(direction, accuracy)
    n_nodes = int(nodes_np.shape[0])
    s_exp = n_nodes * n_coeff

    if s_exp_max is not None and s_exp > int(s_exp_max):
        raise RuntimeError(
            "Yukawa exponential table requires "
            f"S_exp={s_exp}, which exceeds S_exp_max={s_exp_max}."
        )

    operators = [_m2l_operator_for_displacement(node, kappa_h, order) for node in nodes_np]
    operators_np = np.asarray(operators)

    m2e_np = np.tile(np.eye(n_coeff, dtype=np.complex64), (n_nodes, 1))
    e2l_np = np.concatenate(operators_np, axis=1).astype(np.complex64)
    node_weights_np = np.ones((n_nodes,), dtype=np.float32) / max(n_nodes, 1)

    target = _ACCURACY_TARGETS[accuracy]
    validation = validate_yukawa_m2l_exponential_table_arrays(
        nodes_np,
        m2e_np,
        e2l_np,
        operators_np,
    )
    if validation["max_relative_frobenius_error"] > target:
        worst = validation["worst_node_index"]
        raise RuntimeError(
            "Failed to build Yukawa M2L exponential table: "
            f"max_relative_frobenius_error={validation['max_relative_frobenius_error']:.3e}, "
            f"target={target:.3e}, worst_node_index={worst}."
        )

    metadata = {
        "target_relative_error": target,
        "max_relative_frobenius_error": validation["max_relative_frobenius_error"],
        "worst_node_index": validation["worst_node_index"],
        "n_coeff": n_coeff,
        "n_nodes": n_nodes,
        "s_exp": s_exp,
        "axis": _direction_grid(direction, accuracy)[2],
        "sign": _direction_grid(direction, accuracy)[3],
        "other_axes": _direction_grid(direction, accuracy)[4],
        "main_values": tuple(float(x) for x in _direction_grid(direction, accuracy)[0]),
        "transverse_values": tuple(float(x) for x in _direction_grid(direction, accuracy)[1]),
        "representation": "deterministic_operator_table_lagrange",
    }
    return YukawaExponentialM2LTable(
        order=int(order),
        kappa_h=float(kappa_h),
        direction=direction,
        accuracy=accuracy,
        nodes=jnp.asarray(nodes_np, dtype=jnp.float32),
        weights=jnp.asarray(node_weights_np, dtype=jnp.float32),
        m2e=jnp.asarray(m2e_np, dtype=dtype),
        e2l=jnp.asarray(e2l_np, dtype=dtype),
        metadata=metadata,
    )


@lru_cache(maxsize=128)
def _build_cached_table(
    order: int,
    rounded_kappa_h: float,
    direction: str,
    accuracy: str,
    dtype_name: str,
) -> YukawaExponentialM2LTable:
    return _build_raw_table(
        order=order,
        kappa_h=rounded_kappa_h,
        direction=direction,
        accuracy=accuracy,
        dtype_name=dtype_name,
        s_exp_max=None,
    )


def build_yukawa_m2l_exponential_table(
    order: int,
    kappa_h: float,
    direction: str,
    accuracy: str = "1e-3",
    dtype: Any = jnp.complex64,
    s_exp_max: int | None = None,
    require_off_node_accuracy: bool = False,
) -> YukawaExponentialM2LTable:
    """Build or fetch a deterministic operator table for Yukawa M2L.

    ``s_exp_max`` is intentionally not part of the cached path. It is a
    diagnostic guard used by negative tests and exploratory runs.
    """

    rounded_kappa_h = round(float(kappa_h), 8)
    dtype_name = _dtype_name(dtype)
    if s_exp_max is not None or require_off_node_accuracy:
        table = _build_raw_table(
            order=int(order),
            kappa_h=rounded_kappa_h,
            direction=direction,
            accuracy=accuracy,
            dtype_name=dtype_name,
            s_exp_max=None if s_exp_max is None else int(s_exp_max),
        )
        if require_off_node_accuracy:
            validation = validate_yukawa_m2l_exponential_table(table, include_off_node=True)
            table.metadata.update(validation)
            target = _ACCURACY_TARGETS[accuracy]
            if validation["off_node_max_relative_frobenius_error"] > target:
                raise RuntimeError(
                    "Yukawa M2L table failed off-node validation: "
                    "off_node_max_relative_frobenius_error="
                    f"{validation['off_node_max_relative_frobenius_error']:.3e}, "
                    f"target={target:.3e}, "
                    f"worst_off_node_index={validation['worst_off_node_index']}."
                )
        return table
    return _build_cached_table(int(order), rounded_kappa_h, direction, accuracy, dtype_name)


def _lagrange_weights_1d(x: jax.Array, nodes: jax.Array) -> jax.Array:
    x = jnp.asarray(x, dtype=nodes.dtype)
    diff = jnp.abs(x - nodes)
    nearest = jnp.argmin(diff)
    exact = jax.nn.one_hot(nearest, nodes.shape[0], dtype=nodes.dtype)
    denom = nodes[:, None] - nodes[None, :]
    numerator = x - nodes[None, :]
    mask = jnp.eye(nodes.shape[0], dtype=bool)
    ratios = jnp.where(mask, 1.0, numerator / jnp.where(mask, 1.0, denom))
    weights = jnp.prod(ratios, axis=1)
    return jnp.where(diff[nearest] < 1.0e-7, exact, weights)


def _interpolation_weights(table: YukawaExponentialM2LTable, displacement: jax.Array) -> jax.Array:
    displacement = jnp.asarray(displacement, dtype=table.nodes.dtype)
    axis = int(table.metadata["axis"])
    sign = float(table.metadata["sign"])
    other_axes = tuple(int(idx) for idx in table.metadata["other_axes"])
    main_values = jnp.asarray(table.metadata["main_values"], dtype=table.nodes.dtype)
    transverse_values = jnp.asarray(table.metadata["transverse_values"], dtype=table.nodes.dtype)

    main_coord = sign * displacement[axis]
    first_transverse = displacement[other_axes[0]]
    second_transverse = displacement[other_axes[1]]

    main_weights = _lagrange_weights_1d(main_coord, main_values)
    first_weights = _lagrange_weights_1d(first_transverse, transverse_values)
    second_weights = _lagrange_weights_1d(second_transverse, transverse_values)
    return (
        main_weights[:, None, None]
        * first_weights[None, :, None]
        * second_weights[None, None, :]
    ).reshape((-1,))


def apply_exponential_m2l_table(
    table: YukawaExponentialM2LTable,
    source_moments: jax.Array,
    displacement: jax.Array,
) -> jax.Array:
    """Apply the table approximation to one source-moment vector."""

    source_moments = jnp.asarray(source_moments, dtype=table.m2e.dtype)
    n_coeff = (table.order + 1) ** 2
    if source_moments.shape != (n_coeff,):
        raise ValueError(f"source_moments must have shape ({n_coeff},).")
    weights = _interpolation_weights(table, displacement)
    expanded_weights = jnp.repeat(weights.astype(table.m2e.dtype), n_coeff)
    modes = table.m2e @ source_moments
    return table.e2l @ (expanded_weights * modes)


def exponential_m2l_operator_from_table(
    table: YukawaExponentialM2LTable,
    displacement: jax.Array,
) -> jax.Array:
    """Return the approximated M2L matrix represented by ``table``."""

    n_coeff = (table.order + 1) ** 2
    eye = jnp.eye(n_coeff, dtype=table.m2e.dtype)
    return jax.vmap(lambda col: apply_exponential_m2l_table(table, col, displacement), in_axes=1, out_axes=1)(eye)


def validate_yukawa_m2l_exponential_table_arrays(
    nodes: np.ndarray,
    m2e: np.ndarray,
    e2l: np.ndarray,
    reference_operators: np.ndarray,
) -> dict[str, Any]:
    n_nodes = int(nodes.shape[0])
    n_coeff = int(reference_operators.shape[-1])
    errors = []
    for idx in range(n_nodes):
        weights = np.zeros((n_nodes,), dtype=np.complex64)
        weights[idx] = 1.0 + 0.0j
        weighted_m2e = m2e * np.repeat(weights, n_coeff)[:, None]
        approx = e2l @ weighted_m2e
        ref = reference_operators[idx]
        denom = max(float(np.linalg.norm(ref)), 1.0e-30)
        errors.append(float(np.linalg.norm(approx - ref) / denom))
    worst = int(np.argmax(errors)) if errors else -1
    return {
        "max_relative_frobenius_error": float(max(errors) if errors else 0.0),
        "worst_node_index": worst,
        "n_nodes": n_nodes,
    }


def _operator_error(
    approx: np.ndarray,
    reference: np.ndarray,
) -> float:
    denom = max(float(np.linalg.norm(reference)), 1.0e-30)
    return float(np.linalg.norm(approx - reference) / denom)


def validate_yukawa_m2l_exponential_table(
    table: YukawaExponentialM2LTable,
    include_off_node: bool = False,
    test_displacements: np.ndarray | None = None,
) -> dict[str, Any]:
    reference = np.asarray(
        [_m2l_operator_for_displacement(np.asarray(node), table.kappa_h, table.order) for node in table.nodes]
    )
    validation = validate_yukawa_m2l_exponential_table_arrays(
        np.asarray(table.nodes),
        np.asarray(table.m2e),
        np.asarray(table.e2l),
        reference,
    )
    if not include_off_node:
        return validation

    if test_displacements is None:
        test_displacements = _direction_off_node_displacements(table.direction, table.accuracy)
    off_node_errors = []
    for displacement in test_displacements:
        approx = np.asarray(exponential_m2l_operator_from_table(table, displacement))
        ref = _m2l_operator_for_displacement(np.asarray(displacement), table.kappa_h, table.order)
        off_node_errors.append(_operator_error(approx, ref))
    worst = int(np.argmax(off_node_errors)) if off_node_errors else -1
    validation.update(
        {
            "off_node_max_relative_frobenius_error": float(max(off_node_errors) if off_node_errors else 0.0),
            "worst_off_node_index": worst,
            "n_off_node_tests": int(len(off_node_errors)),
            "worst_off_node_displacement": (
                tuple(float(x) for x in test_displacements[worst]) if worst >= 0 else None
            ),
        }
    )
    return validation
