"""Small Yukawa radial-basis utilities used by validation and future FMM work."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnames=("p",))
def modified_spherical_bessel_i(x: jax.Array, p: int) -> jax.Array:
    """Return i_l(x) for l=0..p with stable small-x handling."""

    x = jnp.asarray(x)
    abs_x = jnp.abs(x)
    safe_x = jnp.where(abs_x < 1e-6, 1e-6, x)
    vals = []
    i0 = jnp.sinh(safe_x) / safe_x
    i0_series = 1.0 + x**2 / 6.0 + x**4 / 120.0 + x**6 / 5040.0
    i0 = jnp.where(abs_x < 1e-4, i0_series, i0)
    vals.append(i0)
    if p >= 1:
        i1 = (safe_x * jnp.cosh(safe_x) - jnp.sinh(safe_x)) / safe_x**2
        i1_series = x / 3.0 + x**3 / 30.0 + x**5 / 840.0 + x**7 / 45360.0
        i1 = jnp.where(abs_x < 1e-4, i1_series, i1)
        vals.append(i1)
    for ell in range(1, p):
        recurrence = vals[ell - 1] - (2 * ell + 1) * vals[ell] / safe_x
        denom = 1
        for factor in range(1, 2 * (ell + 1) + 2, 2):
            denom *= factor
        leading = x ** (ell + 1) / float(denom)
        correction = 1.0 + x**2 / (2.0 * (2 * (ell + 1) + 3))
        vals.append(jnp.where(abs_x < 1e-3, leading * correction, recurrence))
    return jnp.stack(vals, axis=-1)


@partial(jax.jit, static_argnames=("p",))
def modified_spherical_bessel_k(x: jax.Array, p: int) -> jax.Array:
    """Return k_l(x) for l=0..p using upward recurrence."""

    x = jnp.asarray(x)
    safe_x = jnp.where(jnp.abs(x) < 1e-12, 1e-12, x)
    vals = []
    exp_term = jnp.exp(-safe_x)
    k0 = 0.5 * jnp.pi * exp_term / safe_x
    vals.append(k0)
    if p >= 1:
        vals.append(k0 * (1.0 + 1.0 / safe_x))
    for ell in range(1, p):
        vals.append(vals[ell - 1] + (2 * ell + 1) * vals[ell] / safe_x)
    return jnp.stack(vals, axis=-1)
