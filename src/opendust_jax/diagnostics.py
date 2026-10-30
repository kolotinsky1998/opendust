"""GPU plane diagnostics and portable NPZ/JSON output."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import platform

import jax
from jax.extend import backend as jax_backend
import jax.numpy as jnp
import numpy as np

from opendust_jax.models import COULOMB_CONSTANT, PlaneDiagnosticConfig
from opendust_jax.physics import background_potential_V


def diagnostic_axes(
    radius_m: float, height_m: float, config: PlaneDiagnosticConfig
) -> tuple[jax.Array, jax.Array]:
    return (
        jnp.linspace(-radius_m, radius_m, config.nx, dtype=jnp.float32),
        jnp.linspace(-0.5 * height_m, 0.5 * height_m, config.nz, dtype=jnp.float32),
    )


def plane_potential_snapshot_V(
    positions_m: jax.Array,
    x_axis_m: jax.Array,
    z_axis_m: jax.Array,
    superion_charge_C: float,
    screening_length_m: float,
    softening_m: float,
    background_coefficients: jax.Array,
    *,
    source_block_size: int = 4096,
    target_block_size: int = 256,
) -> jax.Array:
    """Sum ion Yukawa potentials on y=0 without a full grid-by-ion array."""
    positions = jnp.asarray(positions_m, dtype=jnp.float32)
    xx, zz = jnp.meshgrid(x_axis_m, z_axis_m, indexing="xy")
    targets = jnp.stack((xx.ravel(), jnp.zeros(xx.size), zz.ravel()), axis=1)
    n_sources = positions.shape[0]
    n_targets = targets.shape[0]
    source_blocks = (n_sources + source_block_size - 1) // source_block_size
    target_blocks = (n_targets + target_block_size - 1) // target_block_size
    padded_sources = jnp.pad(
        positions, ((0, source_blocks * source_block_size - n_sources), (0, 0))
    )
    padded_targets = jnp.pad(
        targets, ((0, target_blocks * target_block_size - n_targets), (0, 0))
    )

    def target_body(target_index: int, output: jax.Array) -> jax.Array:
        target_start = target_index * target_block_size
        target = jax.lax.dynamic_slice(
            padded_targets, (target_start, 0), (target_block_size, 3)
        )

        def source_body(source_index: int, partial: jax.Array) -> jax.Array:
            source_start = source_index * source_block_size
            source = jax.lax.dynamic_slice(
                padded_sources, (source_start, 0), (source_block_size, 3)
            )
            valid = source_start + jnp.arange(source_block_size) < n_sources
            displacement = target[:, None, :] - source[None, :, :]
            distance = jnp.sqrt(jnp.sum(displacement * displacement, axis=-1))
            contribution = jnp.exp(-distance / screening_length_m) / (
                distance + softening_m
            )
            return partial + jnp.sum(jnp.where(valid[None, :], contribution, 0.0), axis=1)

        values = jax.lax.fori_loop(
            0, source_blocks, source_body, jnp.zeros(target_block_size, jnp.float32)
        )
        return jax.lax.dynamic_update_slice(output, values, (target_start,))

    ion_sum = jax.lax.fori_loop(
        0,
        target_blocks,
        target_body,
        jnp.zeros(target_blocks * target_block_size, jnp.float32),
    )[:n_targets]
    ion_potential = COULOMB_CONSTANT * superion_charge_C * ion_sum
    background = background_potential_V(
        targets,
        background_coefficients,
        screening_length_m,
        superion_charge_C,
    )
    return (ion_potential + background).reshape((z_axis_m.size, x_axis_m.size))


def plane_cic_density_snapshot_m3(
    positions_m: jax.Array,
    x_axis_m: jax.Array,
    z_axis_m: jax.Array,
    slice_thickness_m: float,
    superion_weight: float,
) -> jax.Array:
    """Deposit a y-slab onto x-z nodes with cloud-in-cell interpolation."""
    positions = jnp.asarray(positions_m, dtype=jnp.float32)
    nx, nz = x_axis_m.size, z_axis_m.size
    dx = x_axis_m[1] - x_axis_m[0]
    dz = z_axis_m[1] - z_axis_m[0]
    gx = (positions[:, 0] - x_axis_m[0]) / dx
    gz = (positions[:, 2] - z_axis_m[0]) / dz
    ix0 = jnp.floor(gx).astype(jnp.int32)
    iz0 = jnp.floor(gz).astype(jnp.int32)
    ix0 = jnp.clip(ix0, 0, nx - 2)
    iz0 = jnp.clip(iz0, 0, nz - 2)
    tx = jnp.clip(gx - ix0, 0.0, 1.0)
    tz = jnp.clip(gz - iz0, 0.0, 1.0)
    selected = jnp.abs(positions[:, 1]) <= 0.5 * slice_thickness_m
    base_weight = jnp.asarray(superion_weight, jnp.float32) * selected
    grid = jnp.zeros((nz, nx), dtype=jnp.float32)
    grid = grid.at[iz0, ix0].add(base_weight * (1.0 - tx) * (1.0 - tz))
    grid = grid.at[iz0, ix0 + 1].add(base_weight * tx * (1.0 - tz))
    grid = grid.at[iz0 + 1, ix0].add(base_weight * (1.0 - tx) * tz)
    grid = grid.at[iz0 + 1, ix0 + 1].add(base_weight * tx * tz)
    return grid / (dx * dz * slice_thickness_m)


def write_plane_diagnostics(
    config: PlaneDiagnosticConfig,
    x_axis_m: np.ndarray,
    z_axis_m: np.ndarray,
    potential_V: np.ndarray,
    ion_density_m3: np.ndarray,
    sample_steps: np.ndarray,
    dt_s: float,
    metadata: dict,
) -> tuple[Path, Path]:
    prefix = Path(config.output_prefix).expanduser()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    npz_path = prefix.with_name(prefix.name + "_fields.npz")
    json_path = prefix.with_name(prefix.name + "_fields.json")
    np.savez_compressed(
        npz_path,
        x_m=np.asarray(x_axis_m),
        z_m=np.asarray(z_axis_m),
        potential_V=np.asarray(potential_V),
        ion_density_m3=np.asarray(ion_density_m3),
        sample_count=np.asarray(len(sample_steps), dtype=np.int64),
        sample_steps=np.asarray(sample_steps, dtype=np.int64),
        sample_times_s=np.asarray(sample_steps, dtype=np.float64) * dt_s,
    )
    document = {
        "format_version": 1,
        "units": {
            "x_m": "m",
            "z_m": "m",
            "potential_V": "V",
            "ion_density_m3": "m^-3",
            "sample_times_s": "s",
        },
        "diagnostic": {**asdict(config), "output_prefix": str(config.output_prefix)},
        "environment": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "jax_backend": jax.default_backend(),
            "jax_platform_version": jax_backend.get_backend().platform_version,
        },
        **metadata,
    }
    json_path.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")
    return npz_path, json_path
