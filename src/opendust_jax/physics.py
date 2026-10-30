"""SI force and charging primitives shared by the simulator and tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from opendust_jax.models import (
    BOLTZMANN_J_K,
    COULOMB_CONSTANT,
    ELECTRON_MASS_KG,
    ELEMENTARY_CHARGE_C,
)


def polynomial_basis(rho: jax.Array, zeta: jax.Array) -> jax.Array:
    """Return the 15-term even cylindrical polynomial basis used by OpenDust."""
    r2 = rho * rho
    z2 = zeta * zeta
    r4, z4 = r2 * r2, z2 * z2
    r6, z6 = r4 * r2, z4 * z2
    return jnp.stack(
        (
            jnp.ones_like(rho), r2, z2, r2 * z2, r4, z4,
            r4 * z2, r2 * z4, r6, z6, r4 * z4,
            r6 * z2, r2 * z6, r4 * r4, z4 * z4,
        ),
        axis=-1,
    )


def polynomial_value_and_gradient(
    positions_m: jax.Array,
    coefficients: jax.Array,
    screening_length_m: float | jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Evaluate dimensionless P and its physical gradient in 1/m."""
    positions = jnp.asarray(positions_m)
    coefficients = jnp.asarray(coefficients, dtype=positions.dtype)
    length = jnp.asarray(screening_length_m, dtype=positions.dtype)
    x, y, z = positions[..., 0], positions[..., 1], positions[..., 2]
    radius = jnp.sqrt(x * x + y * y)
    rho, zeta = radius / length, z / length
    r2, z2 = rho * rho, zeta * zeta
    r3, z3 = r2 * rho, z2 * zeta
    r4, z4 = r2 * r2, z2 * z2
    r5, z5 = r4 * rho, z4 * zeta
    r6, z6 = r4 * r2, z4 * z2
    r7, z7 = r6 * rho, z6 * zeta
    a = coefficients
    value = jnp.sum(polynomial_basis(rho, zeta) * a, axis=-1)
    d_rho = (
        2 * a[1] * rho + 2 * a[3] * rho * z2 + 4 * a[4] * r3
        + 4 * a[6] * r3 * z2 + 2 * a[7] * rho * z4 + 6 * a[8] * r5
        + 4 * a[10] * r3 * z4 + 6 * a[11] * r5 * z2
        + 2 * a[12] * rho * z6 + 8 * a[13] * r7
    )
    d_zeta = (
        2 * a[2] * zeta + 2 * a[3] * r2 * zeta + 4 * a[5] * z3
        + 2 * a[6] * r4 * zeta + 4 * a[7] * r2 * z3 + 6 * a[9] * z5
        + 4 * a[10] * r4 * z3 + 2 * a[11] * r6 * zeta
        + 6 * a[12] * r2 * z5 + 8 * a[14] * z7
    )
    inverse_radius = jnp.where(radius > 0, 1.0 / radius, 0.0)
    gradient = jnp.stack(
        (
            d_rho * x * inverse_radius / length,
            d_rho * y * inverse_radius / length,
            d_zeta / length,
        ),
        axis=-1,
    )
    return value, gradient


def background_potential_V(
    positions_m: jax.Array,
    coefficients: jax.Array,
    screening_length_m: float,
    superion_charge_C: float,
) -> jax.Array:
    value, _ = polynomial_value_and_gradient(
        positions_m, coefficients, screening_length_m
    )
    return COULOMB_CONSTANT * superion_charge_C / screening_length_m * value


def background_force_N(
    positions_m: jax.Array,
    charges_C: float | jax.Array,
    coefficients: jax.Array,
    screening_length_m: float,
    superion_charge_C: float,
) -> jax.Array:
    _, gradient = polynomial_value_and_gradient(
        positions_m, coefficients, screening_length_m
    )
    scale = -COULOMB_CONSTANT * superion_charge_C / screening_length_m
    return scale * jnp.asarray(charges_C)[..., None] * gradient


def ion_dust_coulomb_forces_N(
    ion_positions_m: jax.Array,
    ion_charge_C: float,
    dust_positions_m: jax.Array,
    dust_charges_C: jax.Array,
) -> jax.Array:
    """Bare ion<-dust force from Piel's asymmetric MD model."""
    ions = jnp.asarray(ion_positions_m, dtype=jnp.float32)
    dust_positions = jnp.asarray(dust_positions_m, dtype=jnp.float32)
    dust_charges = jnp.asarray(dust_charges_C, dtype=jnp.float32)
    if dust_positions.shape[0] == 0:
        return jnp.zeros_like(ions)
    displacement = ions[:, None, :] - dust_positions[None, :, :]
    r2 = jnp.sum(displacement * displacement, axis=-1)
    inverse_r = jnp.where(r2 > 0, jax.lax.rsqrt(r2), 0.0)
    coefficient = (
        COULOMB_CONSTANT * ion_charge_C * dust_charges[None, :] * inverse_r**3
    )
    return jnp.sum(coefficient[..., None] * displacement, axis=1)


def dust_orbit_forces_N(
    ion_positions_m: jax.Array,
    ion_charge_C: float,
    dust_positions_m: jax.Array,
    dust_charges_C: jax.Array,
    screening_length_m: float,
) -> jax.Array:
    """Screened dust<-ion force from Piel's asymmetric MD model."""
    ions = jnp.asarray(ion_positions_m, dtype=jnp.float32)
    dust_positions = jnp.asarray(dust_positions_m, dtype=jnp.float32)
    dust_charges = jnp.asarray(dust_charges_C, dtype=jnp.float32)
    if dust_positions.shape[0] == 0:
        return jnp.zeros((0, 3), dtype=jnp.float32)
    displacement = dust_positions[:, None, :] - ions[None, :, :]
    r2 = jnp.sum(displacement * displacement, axis=-1)
    inverse_r = jnp.where(r2 > 0, jax.lax.rsqrt(r2), 0.0)
    distance = jnp.where(r2 > 0, r2 * inverse_r, 1.0)
    coefficient = (
        COULOMB_CONSTANT
        * ion_charge_C
        * dust_charges[:, None]
        * jnp.exp(-distance / screening_length_m)
        * (inverse_r**3 + inverse_r**2 / screening_length_m)
    )
    coefficient = jnp.where(r2 > 0, coefficient, 0.0)
    return jnp.sum(coefficient[..., None] * displacement, axis=1)


def dust_dust_forces_N(
    positions_m: jax.Array, charges_C: jax.Array
) -> jax.Array:
    positions = jnp.asarray(positions_m, dtype=jnp.float32)
    charges = jnp.asarray(charges_C, dtype=jnp.float32)
    count = positions.shape[0]
    if count == 0:
        return jnp.zeros((0, 3), dtype=jnp.float32)
    displacement = positions[:, None, :] - positions[None, :, :]
    r2 = jnp.sum(displacement * displacement, axis=-1)
    mask = ~jnp.eye(count, dtype=bool)
    inverse_r = jnp.where(mask & (r2 > 0), jax.lax.rsqrt(r2), 0.0)
    coefficient = COULOMB_CONSTANT * charges[:, None] * charges[None, :] * inverse_r**3
    return jnp.sum(coefficient[..., None] * displacement, axis=1)


def electron_flux_m2_s(
    dust_charges_C: jax.Array,
    dust_radii_m: jax.Array,
    density_m3: float,
    electron_temperature_K: float,
) -> jax.Array:
    potential = COULOMB_CONSTANT * dust_charges_C / dust_radii_m
    thermal_flux = density_m3 * jnp.sqrt(
        BOLTZMANN_J_K * electron_temperature_K
        / (2.0 * jnp.pi * ELECTRON_MASS_KG)
    )
    return thermal_flux * jnp.exp(
        ELEMENTARY_CHARGE_C * potential
        / (BOLTZMANN_J_K * electron_temperature_K)
    )


def shifted_oml_ion_flux_m2_s(
    dust_charges_C: jax.Array,
    dust_radii_m: jax.Array,
    density_m3: float,
    ion_temperature_K: float,
    ion_mass_kg: float,
    flow_speed_m_s: float,
) -> jax.Array:
    """Shifted-OML ion number flux, including the stable zero-drift limit."""
    v_ti = jnp.sqrt(2.0 * BOLTZMANN_J_K * ion_temperature_K / ion_mass_kg)
    u = jnp.abs(jnp.asarray(flow_speed_m_s, dtype=jnp.float32)) / v_ti
    potential = COULOMB_CONSTANT * dust_charges_C / dust_radii_m
    chi = -ELEMENTARY_CHARGE_C * potential / (
        BOLTZMANN_J_K * ion_temperature_K
    )

    def finite_drift(_: None) -> jax.Array:
        bracket = (
            (1.0 + (0.5 + chi) / (u * u)) * jax.lax.erf(u)
            + jnp.exp(-(u * u)) / (u * jnp.sqrt(jnp.pi))
        )
        return density_m3 * v_ti * u * 0.25 * bracket

    def zero_drift(_: None) -> jax.Array:
        return density_m3 * v_ti / (2.0 * jnp.sqrt(jnp.pi)) * (1.0 + chi)

    return jax.lax.cond(u > 1.0e-4, finite_drift, zero_drift, operand=None)


def update_dust_charges_C(
    charges_C: jax.Array,
    modes: jax.Array,
    absorbed_counts: jax.Array,
    radii_m: jax.Array,
    superion_charge_C: float,
    density_m3: float,
    electron_temperature_K: float,
    ion_temperature_K: float,
    ion_mass_kg: float,
    flow_speed_m_s: float,
    dt_s: float,
) -> jax.Array:
    """Update fixed(0), OML(1), and self-consistent(2) dust charges."""
    electron_flux = electron_flux_m2_s(
        charges_C, radii_m, density_m3, electron_temperature_K
    )
    area = 4.0 * jnp.pi * radii_m * radii_m
    electron_delta = -ELEMENTARY_CHARGE_C * electron_flux * area * dt_s
    ion_flux = shifted_oml_ion_flux_m2_s(
        charges_C,
        radii_m,
        density_m3,
        ion_temperature_K,
        ion_mass_kg,
        flow_speed_m_s,
    )
    oml_delta = electron_delta + ELEMENTARY_CHARGE_C * ion_flux * area * dt_s
    consistent_delta = electron_delta + absorbed_counts * superion_charge_C
    delta = jnp.where(modes == 1, oml_delta, 0.0)
    delta = jnp.where(modes == 2, consistent_delta, delta)
    return charges_C + delta
