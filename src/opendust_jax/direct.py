"""Direct pairwise Coulomb reference solvers in SI units."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp


def _validate_inputs(positions: jax.Array, charges: jax.Array) -> tuple[jax.Array, jax.Array]:
    positions = jnp.asarray(positions)
    charges = jnp.asarray(charges)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3).")
    if charges.ndim != 1:
        raise ValueError("charges must have shape (N,).")
    if positions.shape[0] != charges.shape[0]:
        raise ValueError("positions and charges must contain the same number of particles.")
    return positions, charges


@jax.jit
def _direct_coulomb_field_all_pairs(
    positions: jax.Array, charges: jax.Array, eps0: float
) -> jax.Array:
    diff = positions[:, None, :] - positions[None, :, :]
    r = jnp.linalg.norm(diff, axis=-1)
    inv_r3 = jnp.where(r == 0.0, 0.0, 1.0 / (r**3))
    field = jnp.sum(charges[None, :, None] * diff * inv_r3[:, :, None], axis=1)
    return field / (4.0 * jnp.pi * eps0)


@partial(jax.jit, static_argnames=("batch_size",))
def _direct_coulomb_field_batched(
    positions: jax.Array, charges: jax.Array, eps0: float, batch_size: int
) -> jax.Array:
    n_particles = positions.shape[0]
    n_batches = (n_particles + batch_size - 1) // batch_size
    padded_n = n_batches * batch_size
    pad = padded_n - n_particles
    padded_positions = jnp.pad(positions, ((0, pad), (0, 0)))
    batch_starts = jnp.arange(n_batches) * batch_size

    def batch_field(start: jax.Array) -> jax.Array:
        eval_positions = jax.lax.dynamic_slice(padded_positions, (start, 0), (batch_size, 3))
        diff = eval_positions[:, None, :] - positions[None, :, :]
        r = jnp.linalg.norm(diff, axis=-1)
        inv_r3 = jnp.where(r == 0.0, 0.0, 1.0 / (r**3))
        return jnp.sum(charges[None, :, None] * diff * inv_r3[:, :, None], axis=1)

    def scan_body(field_buffer: jax.Array, start: jax.Array) -> tuple[jax.Array, None]:
        field_chunk = batch_field(start)
        field_buffer = jax.lax.dynamic_update_slice(field_buffer, field_chunk, (start, 0))
        return field_buffer, None

    field_dtype = jnp.result_type(positions, charges, jnp.asarray(eps0))
    field, _ = jax.lax.scan(scan_body, jnp.zeros((padded_n, 3), field_dtype), batch_starts)
    field = field[:n_particles]
    return field / (4.0 * jnp.pi * eps0)


def direct_coulomb_field(
    positions: jax.Array,
    charges: jax.Array,
    eps0: float = 8.85418781762039e-12,
    batch_size: int | None = None,
) -> jax.Array:
    """Evaluate the free-space Coulomb electric field by direct pairwise summation."""

    positions, charges = _validate_inputs(positions, charges)
    if eps0 <= 0:
        raise ValueError("eps0 must be positive.")
    if batch_size is None:
        return _direct_coulomb_field_all_pairs(positions, charges, eps0)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive when provided.")
    return _direct_coulomb_field_batched(positions, charges, eps0, int(batch_size))


def direct_coulomb_forces(
    positions: jax.Array,
    charges: jax.Array,
    eps0: float = 8.85418781762039e-12,
    batch_size: int | None = None,
) -> jax.Array:
    """Evaluate direct Coulomb forces on each particle in newtons."""

    positions, charges = _validate_inputs(positions, charges)
    field = direct_coulomb_field(positions, charges, eps0=eps0, batch_size=batch_size)
    return charges[:, None] * field
