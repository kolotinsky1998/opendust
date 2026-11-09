"""Device-portable blocked Yukawa potential sums."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def blocked_yukawa_sum(
    sources_m: jax.Array,
    targets_m: jax.Array,
    screening_length_m: float | jax.Array,
    softening_m: float | jax.Array,
    *,
    source_block_size: int,
    target_block_size: int = 256,
) -> jax.Array:
    """Sum ``exp(-r/lambda) / (r + softening)`` at each target.

    Both block sizes are intended to be static while this function is traced.
    Padding keeps every loop iteration the same shape without materializing the
    complete target-by-source displacement array.
    """
    sources = jnp.asarray(sources_m, dtype=jnp.float32)
    targets = jnp.asarray(targets_m, dtype=jnp.float32)
    screening_length = jnp.asarray(screening_length_m, dtype=jnp.float32)
    softening = jnp.asarray(softening_m, dtype=jnp.float32)
    n_sources = sources.shape[0]
    n_targets = targets.shape[0]
    source_blocks = (n_sources + source_block_size - 1) // source_block_size
    target_blocks = (n_targets + target_block_size - 1) // target_block_size
    padded_sources = jnp.pad(
        sources, ((0, source_blocks * source_block_size - n_sources), (0, 0))
    )
    padded_targets = jnp.pad(
        targets, ((0, target_blocks * target_block_size - n_targets), (0, 0))
    )

    def target_body(target_index: int, output: jax.Array) -> jax.Array:
        target_start = target_index * target_block_size
        target_zero = jnp.asarray(0, dtype=target_start.dtype)
        target = jax.lax.dynamic_slice(
            padded_targets, (target_start, target_zero), (target_block_size, 3)
        )

        def source_body(source_index: int, partial: jax.Array) -> jax.Array:
            source_start = source_index * source_block_size
            source_zero = jnp.asarray(0, dtype=source_start.dtype)
            source = jax.lax.dynamic_slice(
                padded_sources,
                (source_start, source_zero),
                (source_block_size, 3),
            )
            valid = source_start + jnp.arange(source_block_size) < n_sources
            displacement = target[:, None, :] - source[None, :, :]
            distance = jnp.sqrt(jnp.sum(displacement * displacement, axis=-1))
            contribution = jnp.exp(-distance / screening_length) / (
                distance + softening
            )
            return partial + jnp.sum(
                jnp.where(valid[None, :], contribution, 0.0), axis=1
            )

        values = jax.lax.fori_loop(
            0,
            source_blocks,
            source_body,
            jnp.zeros(target_block_size, dtype=jnp.float32),
        )
        return jax.lax.dynamic_update_slice(output, values, (target_start,))

    return jax.lax.fori_loop(
        0,
        target_blocks,
        target_body,
        jnp.zeros(target_blocks * target_block_size, dtype=jnp.float32),
    )[:n_targets]


__all__ = ["blocked_yukawa_sum"]
