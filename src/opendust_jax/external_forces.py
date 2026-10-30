"""Fused and reference implementations of non-pair ion forces."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from opendust_jax.cuda_pair_forces import external_forces_cuda
from opendust_jax.physics import (
    background_force_N,
    dust_orbit_forces_N,
    ion_dust_coulomb_forces_N,
)


def external_forces_jax(
    positions_m: jax.Array,
    dust_positions_m: jax.Array,
    dust_charges_C: jax.Array,
    background_coefficients: jax.Array,
    ion_charge_C: float,
    screening_length_m: float,
    electric_field_V_m: float,
) -> tuple[jax.Array, jax.Array]:
    forces = ion_dust_coulomb_forces_N(
        positions_m, ion_charge_C, dust_positions_m, dust_charges_C
    )
    forces = forces + background_force_N(
        positions_m,
        ion_charge_C,
        background_coefficients,
        screening_length_m,
        ion_charge_C,
    )
    field = jnp.asarray((0.0, 0.0, ion_charge_C * electric_field_V_m), jnp.float32)
    forces = forces + field
    orbit = dust_orbit_forces_N(
        positions_m,
        ion_charge_C,
        dust_positions_m,
        dust_charges_C,
        screening_length_m,
    )
    return forces, orbit


__all__ = ["external_forces_cuda", "external_forces_jax"]
