"""Validation metrics for comparing direct and FMM force evaluations."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class ValidationMetrics:
    relative_l2: float
    relative_l2_x: float
    relative_l2_y: float
    relative_l2_z: float
    max_abs_error: float
    max_relative_error: float
    pearson_correlation: float
    slope: float
    intercept: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _relative_l2(actual: jax.Array, reference: jax.Array) -> jax.Array:
    denom = jnp.linalg.norm(reference)
    return jnp.where(denom == 0.0, jnp.linalg.norm(actual - reference), jnp.linalg.norm(actual - reference) / denom)


def compute_force_metrics(
    reference_forces: jax.Array,
    computed_forces: jax.Array,
    relative_floor: float = 1e-30,
) -> ValidationMetrics:
    """Compute scalar diagnostics for FMM forces against direct reference forces."""

    reference_forces = jnp.asarray(reference_forces)
    computed_forces = jnp.asarray(computed_forces)
    if reference_forces.shape != computed_forces.shape:
        raise ValueError("reference_forces and computed_forces must have the same shape.")
    if reference_forces.ndim != 2 or reference_forces.shape[1] != 3:
        raise ValueError("forces must have shape (N, 3).")
    if relative_floor <= 0:
        raise ValueError("relative_floor must be positive.")

    error = computed_forces - reference_forces
    flat_ref = reference_forces.reshape(-1)
    flat_computed = computed_forces.reshape(-1)
    flat_error = error.reshape(-1)
    mask = jnp.abs(flat_ref) > relative_floor

    ref_centered = flat_ref - jnp.mean(flat_ref)
    computed_centered = flat_computed - jnp.mean(flat_computed)
    corr_denom = jnp.linalg.norm(ref_centered) * jnp.linalg.norm(computed_centered)
    slope_denom = jnp.sum(ref_centered**2)
    slope = jnp.where(slope_denom == 0.0, 0.0, jnp.sum(ref_centered * computed_centered) / slope_denom)
    intercept = jnp.mean(flat_computed) - slope * jnp.mean(flat_ref)
    safe_reference = jnp.where(mask, flat_ref, 1.0)
    relative_error_values = jnp.where(mask, jnp.abs(flat_error / safe_reference), 0.0)
    max_relative_error = jnp.where(jnp.any(mask), jnp.max(relative_error_values), 0.0)

    return ValidationMetrics(
        relative_l2=float(_relative_l2(computed_forces, reference_forces)),
        relative_l2_x=float(_relative_l2(computed_forces[:, 0], reference_forces[:, 0])),
        relative_l2_y=float(_relative_l2(computed_forces[:, 1], reference_forces[:, 1])),
        relative_l2_z=float(_relative_l2(computed_forces[:, 2], reference_forces[:, 2])),
        max_abs_error=float(jnp.max(jnp.abs(error))),
        max_relative_error=float(max_relative_error),
        pearson_correlation=float(jnp.where(corr_denom == 0.0, 0.0, jnp.sum(ref_centered * computed_centered) / corr_denom)),
        slope=float(slope),
        intercept=float(intercept),
    )
