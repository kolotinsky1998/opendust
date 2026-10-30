"""Deterministic polynomial approximation of the OpenDust confining background."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from opendust_jax.models import PolynomialBackgroundConfig


@dataclass(frozen=True)
class PolynomialBackground:
    coefficients: np.ndarray
    relative_rms_error: float
    sample_count: int
    cache_key: str


def _basis_numpy(rho: np.ndarray, zeta: np.ndarray) -> np.ndarray:
    r2, z2 = rho * rho, zeta * zeta
    r4, z4 = r2 * r2, z2 * z2
    r6, z6 = r4 * r2, z4 * z2
    return np.stack(
        (
            np.ones_like(rho), r2, z2, r2 * z2, r4, z4,
            r4 * z2, r2 * z4, r6, z6, r4 * z4,
            r6 * z2, r2 * z6, r4 * r4, z4 * z4,
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
        "version": 1,
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
        return PolynomialBackground(np.zeros(15), 0.0, 0, key)

    cache_path: Path | None = None
    if config.cache_dir is not None:
        cache_dir = Path(config.cache_dir).expanduser()
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"background_{key}.npz"
        if cache_path.is_file():
            data = np.load(cache_path)
            return PolynomialBackground(
                np.asarray(data["coefficients"], dtype=np.float64),
                float(data["relative_rms_error"]),
                int(data["sample_count"]),
                key,
            )

    radius = radius_m / screening_length_m
    height = height_m / screening_length_m
    radial_count = max(8, int(np.ceil(radius * config.radial_points_per_screening_length)) + 1)
    axial_count = max(8, int(np.ceil(0.5 * height * config.axial_points_per_screening_length)) + 1)
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
    potential = np.zeros(targets.shape[0], dtype=np.float64)
    for start in range(0, config.sample_count, config.source_block_size):
        block = sources[start : start + config.source_block_size]
        displacement = targets[:, None, :] - block[None, :, :]
        distance = np.linalg.norm(displacement, axis=-1)
        potential += np.sum(np.exp(-distance) / (distance + 1.0e-3), axis=1)
    potential /= config.sample_count

    design = _basis_numpy(rr.ravel(), zz.ravel())
    normalized_coefficients, *_ = np.linalg.lstsq(design, potential, rcond=None)
    fitted = design @ normalized_coefficients
    residual = np.linalg.norm(fitted - potential)
    relative_rms = float(residual / max(np.linalg.norm(potential), np.finfo(float).tiny))
    coefficients = -normalized_coefficients * float(particle_count)

    if cache_path is not None:
        np.savez_compressed(
            cache_path,
            coefficients=coefficients,
            relative_rms_error=np.asarray(relative_rms),
            sample_count=np.asarray(config.sample_count),
        )
    return PolynomialBackground(coefficients, relative_rms, config.sample_count, key)
