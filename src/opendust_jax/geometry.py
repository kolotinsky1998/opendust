"""Geometry helpers for cylindrical OpenDust validation problems."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class CylinderDomain:
    """Cylindrical source domain in SI units."""

    R: float
    H: float

    def __post_init__(self) -> None:
        if self.R <= 0:
            raise ValueError("Cylinder radius R must be positive.")
        if self.H <= 0:
            raise ValueError("Cylinder height H must be positive.")


def sample_uniform_cylinder(domain: CylinderDomain, n: int, seed: int = 0) -> jax.Array:
    """Sample points uniformly inside a cylinder centered at the origin.

    Coordinates are returned in SI units. The cylinder axis is the z-axis and
    z lies in [-H/2, H/2].
    """

    if n < 0:
        raise ValueError("Number of points n must be non-negative.")

    key = jax.random.PRNGKey(seed)
    key_r, key_theta, key_z = jax.random.split(key, 3)
    radial = domain.R * jnp.sqrt(jax.random.uniform(key_r, (n,)))
    theta = 2.0 * jnp.pi * jax.random.uniform(key_theta, (n,))
    z = domain.H * (jax.random.uniform(key_z, (n,)) - 0.5)
    x = radial * jnp.cos(theta)
    y = radial * jnp.sin(theta)
    return jnp.stack((x, y, z), axis=1)


def points_inside_cylinder(
    points: jax.Array, domain: CylinderDomain, tolerance: float = 1e-12
) -> jax.Array:
    """Return a boolean mask for points inside the closed cylinder."""

    points = jnp.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3).")

    radial = jnp.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
    inside_radial = radial <= domain.R + tolerance
    inside_height = jnp.abs(points[:, 2]) <= 0.5 * domain.H + tolerance
    return inside_radial & inside_height
