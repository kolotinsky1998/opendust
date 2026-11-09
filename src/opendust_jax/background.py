"""Deterministic polynomial approximation of the OpenDust confining background."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from opendust_jax.grid_potential import blocked_yukawa_sum
from opendust_jax.models import PolynomialBackgroundConfig


@dataclass(frozen=True)
class PolynomialBackground:
    coefficients: np.ndarray
    relative_rms_error: float
    sample_count: int
    cache_key: str
    _device_coefficients: jax.Array | None = field(
        default=None, init=False, repr=False, compare=False
    )

    @property
    def device_coefficients(self) -> jax.Array:
        """Coefficients already resident on the active JAX device when possible."""
        if self._device_coefficients is not None:
            return self._device_coefficients
        return jnp.asarray(self.coefficients, dtype=jnp.float32)


def _attach_device_coefficients(
    background: PolynomialBackground, coefficients: jax.Array
) -> PolynomialBackground:
    object.__setattr__(background, "_device_coefficients", coefficients)
    return background


def _basis_jax(rho: jax.Array, zeta: jax.Array) -> jax.Array:
    r2, z2 = rho * rho, zeta * zeta
    r4, z4 = r2 * r2, z2 * z2
    r6, z6 = r4 * r2, z4 * z2
    return jnp.stack(
        (
            jnp.ones_like(rho),
            r2,
            z2,
            r2 * z2,
            r4,
            z4,
            r4 * z2,
            r2 * z4,
            r6,
            z6,
            r4 * z4,
            r6 * z2,
            r2 * z6,
            r4 * r4,
            z4 * z4,
        ),
        axis=-1,
    )


def _cache_key(
    radius_m: float,
    height_m: float,
    screening_length_m: float,
    particle_count: int,
    config: PolynomialBackgroundConfig,
) -> str:
    payload = {
        "version": 2,
        "radius_m": radius_m,
        "height_m": height_m,
        "screening_length_m": screening_length_m,
        "particle_count": particle_count,
        "config": {**asdict(config), "cache_dir": None},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:20]


def build_polynomial_background(
    radius_m: float,
    height_m: float,
    screening_length_m: float,
    particle_count: int,
    config: PolynomialBackgroundConfig,
) -> PolynomialBackground:
    """Fit the legacy 15-term cavity potential with deterministic samples."""
    key = _cache_key(
        radius_m, height_m, screening_length_m, particle_count, config
    )
    if not config.enabled:
        coefficients = np.zeros(15)
        return _attach_device_coefficients(
            PolynomialBackground(coefficients, 0.0, 0, key),
            jnp.asarray(coefficients, dtype=jnp.float32),
        )

    cache_path: Path | None = None
    if config.cache_dir is not None:
        cache_dir = Path(config.cache_dir).expanduser()
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"background_{key}.npz"
        if cache_path.is_file():
            data = np.load(cache_path)
            coefficients = np.asarray(data["coefficients"], dtype=np.float64)
            return _attach_device_coefficients(
                PolynomialBackground(
                    coefficients,
                    float(data["relative_rms_error"]),
                    int(data["sample_count"]),
                    key,
                ),
                jnp.asarray(coefficients, dtype=jnp.float32),
            )

    radius = radius_m / screening_length_m
    height = height_m / screening_length_m
    radial_count = max(
        8, int(np.ceil(radius * config.radial_points_per_screening_length)) + 1
    )
    axial_count = max(
        8,
        int(np.ceil(0.5 * height * config.axial_points_per_screening_length)) + 1,
    )
    grid_r = np.linspace(0.0, radius, radial_count)
    grid_z = np.linspace(0.0, 0.5 * height, axial_count)
    zz, rr = np.meshgrid(grid_z, grid_r, indexing="xy")
    targets = np.stack((rr.ravel(), np.zeros(rr.size), zz.ravel()), axis=1)

    rng = np.random.default_rng(config.seed)
    source_r = radius * np.sqrt(rng.random(config.sample_count))
    source_theta = 2.0 * np.pi * rng.random(config.sample_count)
    sources = np.stack(
        (
            source_r * np.cos(source_theta),
            source_r * np.sin(source_theta),
            height * (rng.random(config.sample_count) - 0.5),
        ),
        axis=1,
    )
    device_sources, device_targets = jax.device_put(
        (
            sources.astype(np.float32, copy=False),
            targets.astype(np.float32, copy=False),
        )
    )

    def fit_on_device(source_positions: jax.Array, target_positions: jax.Array):
        potential = blocked_yukawa_sum(
            source_positions,
            target_positions,
            1.0,
            1.0e-3,
            source_block_size=config.source_block_size,
        ) / jnp.asarray(config.sample_count, dtype=jnp.float32)
        design = _basis_jax(target_positions[:, 0], target_positions[:, 2])
        normalized_coefficients, *_ = jnp.linalg.lstsq(design, potential, rcond=None)
        fitted = design @ normalized_coefficients
        residual = jnp.linalg.norm(fitted - potential)
        scale = jnp.maximum(jnp.linalg.norm(potential), jnp.finfo(jnp.float32).tiny)
        coefficients = -normalized_coefficients * jnp.asarray(
            particle_count, dtype=jnp.float32
        )
        return coefficients, residual / scale

    device_coefficients, device_relative_rms = jax.jit(fit_on_device)(
        device_sources, device_targets
    )
    host_coefficients, host_relative_rms = jax.device_get(
        (device_coefficients, device_relative_rms)
    )
    coefficients = np.asarray(host_coefficients, dtype=np.float64)
    relative_rms = float(host_relative_rms)

    if cache_path is not None:
        np.savez_compressed(
            cache_path,
            coefficients=coefficients,
            relative_rms_error=np.asarray(relative_rms),
            sample_count=np.asarray(config.sample_count),
        )
    return _attach_device_coefficients(
        PolynomialBackground(coefficients, relative_rms, config.sample_count, key),
        device_coefficients,
    )
