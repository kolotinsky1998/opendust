"""Direct JAX pair forces for the softened Yukawa potential used by OpenDust."""

from __future__ import annotations

from functools import partial
from numbers import Real

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnames=("block_size",))
def _softened_yukawa_forces_tiled(
    positions: jax.Array,
    prefactor: jax.Array,
    screening_length: jax.Array,
    softening: jax.Array,
    *,
    block_size: int,
) -> jax.Array:
    """Evaluate target blocks without materializing an (N, N, 3) array."""

    n_particles = positions.shape[0]
    n_blocks = (n_particles + block_size - 1) // block_size
    padded_n = n_blocks * block_size
    padded_positions = jnp.pad(positions, ((0, padded_n - n_particles), (0, 0)))
    source_indices = jnp.arange(n_particles, dtype=jnp.int32)[None, :]
    target_offsets = jnp.arange(block_size, dtype=jnp.int32)[:, None]
    block_starts = jnp.arange(n_blocks, dtype=jnp.int32) * block_size
    inverse_screening_length = jnp.reciprocal(screening_length)

    def evaluate_block(start: jax.Array) -> jax.Array:
        targets = jax.lax.dynamic_slice(
            padded_positions, (start, 0), (block_size, 3)
        )
        target_indices = start + target_offsets
        pair_is_valid = (target_indices < n_particles) & (
            target_indices != source_indices
        )

        displacement = targets[:, None, :] - positions[None, :, :]
        distance_squared = jnp.sum(displacement * displacement, axis=-1)

        # Self and padded pairs use a finite dummy distance. Masking only after
        # division would allow NaNs to enter the compiled computation.
        safe_distance_squared = jnp.where(pair_is_valid, distance_squared, 1.0)
        inverse_distance = jax.lax.rsqrt(safe_distance_squared)
        distance = safe_distance_squared * inverse_distance
        denominator = distance + softening
        radial_coefficient = (
            prefactor
            * jnp.exp(-distance * inverse_screening_length)
            * (
                inverse_screening_length / denominator
                + jnp.reciprocal(denominator * denominator)
            )
            * inverse_distance
        )
        radial_coefficient = jnp.where(pair_is_valid, radial_coefficient, 0.0)
        return jnp.sum(radial_coefficient[..., None] * displacement, axis=1)

    def store_block(
        force_buffer: jax.Array, start: jax.Array
    ) -> tuple[jax.Array, None]:
        block_forces = evaluate_block(start)
        force_buffer = jax.lax.dynamic_update_slice(
            force_buffer, block_forces, (start, 0)
        )
        return force_buffer, None

    force_buffer, _ = jax.lax.scan(
        store_block,
        jnp.zeros((padded_n, 3), dtype=jnp.float32),
        block_starts,
    )
    return force_buffer[:n_particles]


@partial(
    jax.jit,
    static_argnames=("target_block_size", "source_block_size"),
)
def _softened_yukawa_forces_2d_tiled(
    positions: jax.Array,
    prefactor: jax.Array,
    screening_length: jax.Array,
    softening: jax.Array,
    *,
    target_block_size: int,
    source_block_size: int,
) -> jax.Array:
    """Evaluate fixed-size target/source tiles and accumulate target forces."""

    n_particles = positions.shape[0]
    n_target_blocks = (
        n_particles + target_block_size - 1
    ) // target_block_size
    n_source_blocks = (
        n_particles + source_block_size - 1
    ) // source_block_size
    padded_target_n = n_target_blocks * target_block_size
    padded_source_n = n_source_blocks * source_block_size
    target_positions = jnp.pad(
        positions, ((0, padded_target_n - n_particles), (0, 0))
    )
    source_positions = jnp.pad(
        positions, ((0, padded_source_n - n_particles), (0, 0))
    )
    target_offsets = jnp.arange(target_block_size, dtype=jnp.int32)
    source_offsets = jnp.arange(source_block_size, dtype=jnp.int32)
    target_starts = (
        jnp.arange(n_target_blocks, dtype=jnp.int32) * target_block_size
    )
    inverse_screening_length = jnp.reciprocal(screening_length)

    def evaluate_target_block(target_start: jax.Array) -> jax.Array:
        targets = jax.lax.dynamic_slice(
            target_positions,
            (target_start, 0),
            (target_block_size, 3),
        )
        target_indices = target_start + target_offsets
        target_is_valid = target_indices < n_particles
        target_x = targets[:, 0, None]
        target_y = targets[:, 1, None]
        target_z = targets[:, 2, None]

        def accumulate_source_block(
            source_block: int, accumulated: jax.Array
        ) -> jax.Array:
            source_start = source_block * source_block_size
            sources = jax.lax.dynamic_slice(
                source_positions,
                (source_start, 0),
                (source_block_size, 3),
            )
            source_indices = source_start + source_offsets
            pair_is_valid = (
                target_is_valid[:, None]
                & (source_indices[None, :] < n_particles)
                & (target_indices[:, None] != source_indices[None, :])
            )

            dx = target_x - sources[None, :, 0]
            dy = target_y - sources[None, :, 1]
            dz = target_z - sources[None, :, 2]
            distance_squared = dx * dx + dy * dy + dz * dz
            safe_distance_squared = jnp.where(
                pair_is_valid, distance_squared, 1.0
            )
            inverse_distance = jax.lax.rsqrt(safe_distance_squared)
            distance = safe_distance_squared * inverse_distance
            denominator = distance + softening
            radial_coefficient = (
                prefactor
                * jnp.exp(-distance * inverse_screening_length)
                * (
                    inverse_screening_length / denominator
                    + jnp.reciprocal(denominator * denominator)
                )
                * inverse_distance
            )
            radial_coefficient = jnp.where(
                pair_is_valid, radial_coefficient, 0.0
            )
            tile_force = jnp.stack(
                (
                    jnp.sum(radial_coefficient * dx, axis=1),
                    jnp.sum(radial_coefficient * dy, axis=1),
                    jnp.sum(radial_coefficient * dz, axis=1),
                ),
                axis=1,
            )
            return accumulated + tile_force

        return jax.lax.fori_loop(
            0,
            n_source_blocks,
            accumulate_source_block,
            jnp.zeros((target_block_size, 3), dtype=jnp.float32),
        )

    def store_target_block(
        force_buffer: jax.Array, target_start: jax.Array
    ) -> tuple[jax.Array, None]:
        block_forces = evaluate_target_block(target_start)
        force_buffer = jax.lax.dynamic_update_slice(
            force_buffer, block_forces, (target_start, 0)
        )
        return force_buffer, None

    force_buffer, _ = jax.lax.scan(
        store_target_block,
        jnp.zeros((padded_target_n, 3), dtype=jnp.float32),
        target_starts,
    )
    return force_buffer[:n_particles]


def softened_yukawa_forces(
    positions: jax.Array,
    prefactor: float | jax.Array,
    screening_length: float | jax.Array,
    softening: float | jax.Array,
    *,
    block_size: int = 128,
) -> jax.Array:
    """Calculate exact O(N^2) forces for OpenDust's softened Yukawa potential.

    The evaluated pair potential is

    ``U(r) = prefactor * exp(-r / screening_length) / (r + softening)``.

    All inputs are converted to FP32 to match the current OpenMM CUDA setup.
    ``block_size`` controls the number of targets processed in one XLA tile;
    it is static and changing it triggers a separate compilation.

    Args:
        positions: Particle coordinates with shape ``(N, 3)``.
        prefactor: Complete multiplicative coefficient in the potential.
        screening_length: Positive screening (Debye) length.
        softening: Non-negative distance offset in the denominator.
        block_size: Positive target tile size.

    Returns:
        FP32 forces with shape ``(N, 3)`` in units consistent with the inputs.
    """

    positions = jnp.asarray(positions, dtype=jnp.float32)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3).")
    if positions.shape[0] == 0:
        raise ValueError("positions must contain at least one particle.")
    if not isinstance(block_size, int) or isinstance(block_size, bool) or block_size <= 0:
        raise ValueError("block_size must be a positive integer.")
    if isinstance(screening_length, Real) and screening_length <= 0.0:
        raise ValueError("screening_length must be positive.")
    if isinstance(softening, Real) and softening < 0.0:
        raise ValueError("softening must be non-negative.")

    prefactor = jnp.asarray(prefactor, dtype=jnp.float32)
    screening_length = jnp.asarray(screening_length, dtype=jnp.float32)
    softening = jnp.asarray(softening, dtype=jnp.float32)
    for name, value in (
        ("prefactor", prefactor),
        ("screening_length", screening_length),
        ("softening", softening),
    ):
        if value.ndim != 0:
            raise ValueError(f"{name} must be a scalar.")

    return _softened_yukawa_forces_tiled(
        positions,
        prefactor,
        screening_length,
        softening,
        block_size=block_size,
    )


def softened_yukawa_forces_2d(
    positions: jax.Array,
    prefactor: float | jax.Array,
    screening_length: float | jax.Array,
    softening: float | jax.Array,
    *,
    target_block_size: int = 256,
    source_block_size: int = 256,
) -> jax.Array:
    """Calculate exact forces using separate target and source tile sizes.

    This backend preserves the same O(N**2) model and FP32 output as
    :func:`softened_yukawa_forces`, but bounds pairwise intermediates by
    ``(target_block_size, source_block_size)``. Source tiles are accumulated
    with ``jax.lax.fori_loop``.
    """

    positions = jnp.asarray(positions, dtype=jnp.float32)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3).")
    if positions.shape[0] == 0:
        raise ValueError("positions must contain at least one particle.")
    for name, value in (
        ("target_block_size", target_block_size),
        ("source_block_size", source_block_size),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if isinstance(screening_length, Real) and screening_length <= 0.0:
        raise ValueError("screening_length must be positive.")
    if isinstance(softening, Real) and softening < 0.0:
        raise ValueError("softening must be non-negative.")

    prefactor = jnp.asarray(prefactor, dtype=jnp.float32)
    screening_length = jnp.asarray(screening_length, dtype=jnp.float32)
    softening = jnp.asarray(softening, dtype=jnp.float32)
    for name, value in (
        ("prefactor", prefactor),
        ("screening_length", screening_length),
        ("softening", softening),
    ):
        if value.ndim != 0:
            raise ValueError(f"{name} must be a scalar.")

    return _softened_yukawa_forces_2d_tiled(
        positions,
        prefactor,
        screening_length,
        softening,
        target_block_size=target_block_size,
        source_block_size=source_block_size,
    )
