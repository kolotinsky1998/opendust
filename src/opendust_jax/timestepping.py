"""Time-step recommendations for the physical main integration step."""

from __future__ import annotations

from dataclasses import dataclass
import math

from opendust_jax.collisions import ArgonNullCollisionModel
from opendust_jax.models import (
    ArgonPresheathPlasmaSI,
    ELEMENTARY_CHARGE_C,
    EPSILON_0_F_M,
    FieldDrivenPlasmaSI,
    PlasmaSI,
)


@dataclass(frozen=True)
class TimeStepRecommendation:
    ion_plasma_frequency_s: float
    collision_frequency_s: float | None
    plasma_limit_s: float
    collision_limit_s: float | None
    recommended_dt_s: float


def recommend_main_timestep(
    plasma: PlasmaSI,
    collision_model: ArgonNullCollisionModel | None = None,
) -> TimeStepRecommendation:
    """Return the main-step limits from ion plasma and neutral collisions."""

    omega_pi = math.sqrt(
        plasma.density_m3 * ELEMENTARY_CHARGE_C**2
        / (EPSILON_0_F_M * plasma.ion_mass_kg)
    )
    plasma_limit = 0.1 / omega_pi
    collision_frequency: float | None = None
    collision_limit: float | None = None
    if isinstance(plasma, FieldDrivenPlasmaSI):
        collision_frequency = plasma.collision_frequency_s
        collision_limit = 0.1 / collision_frequency
    elif isinstance(plasma, ArgonPresheathPlasmaSI):
        model = collision_model or ArgonNullCollisionModel(plasma)
        collision_frequency = model.maximum_frequency_s
        collision_limit = -math.log(0.9) / collision_frequency
    recommended = min(
        plasma_limit,
        math.inf if collision_limit is None else collision_limit,
    )
    return TimeStepRecommendation(
        ion_plasma_frequency_s=omega_pi,
        collision_frequency_s=collision_frequency,
        plasma_limit_s=plasma_limit,
        collision_limit_s=collision_limit,
        recommended_dt_s=recommended,
    )


__all__ = ["TimeStepRecommendation", "recommend_main_timestep"]
