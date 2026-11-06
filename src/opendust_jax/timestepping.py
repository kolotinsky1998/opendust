"""Time-step recommendations for the physical main integration step."""

from __future__ import annotations

from dataclasses import dataclass
import math

from opendust_jax.collisions import ArgonNullCollisionModel
from opendust_jax.models import (
    ArgonPresheathPlasmaSI,
    CollisionSubcyclingConfig,
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
    collision_substeps: int = 1
    collision_dt_s: float | None = None


def collision_timestep_limit_s(
    plasma: PlasmaSI,
    maximum_candidate_probability: float,
    collision_model: ArgonNullCollisionModel | None = None,
) -> tuple[float | None, float | None]:
    """Return collision frequency and the largest Bernoulli sampling step."""
    if isinstance(plasma, FieldDrivenPlasmaSI):
        return (
            plasma.collision_frequency_s,
            maximum_candidate_probability / plasma.collision_frequency_s,
        )
    if isinstance(plasma, ArgonPresheathPlasmaSI):
        model = collision_model or ArgonNullCollisionModel(plasma)
        probability = min(
            maximum_candidate_probability, model.maximum_candidate_probability
        )
        return (
            model.maximum_frequency_s,
            -math.log1p(-probability) / model.maximum_frequency_s,
        )
    return None, None


def resolve_collision_substeps(
    plasma: PlasmaSI,
    dt_s: float,
    config: CollisionSubcyclingConfig | None,
    collision_model: ArgonNullCollisionModel | None = None,
) -> tuple[int, float, float | None, float | None]:
    """Resolve collision substeps and reject an unsafe explicit override."""
    if config is None:
        probability = (
            collision_model.maximum_candidate_probability
            if isinstance(plasma, ArgonPresheathPlasmaSI)
            and collision_model is not None
            else 0.1
        )
        frequency, limit = collision_timestep_limit_s(
            plasma, probability, collision_model
        )
        return 1, dt_s, frequency, limit
    frequency, limit = collision_timestep_limit_s(
        plasma, config.maximum_candidate_probability, collision_model
    )
    minimum = 1 if limit is None else max(1, math.ceil(dt_s / limit))
    resolved = minimum if config.substeps is None else config.substeps
    if resolved < minimum:
        raise ValueError(
            f"collision substeps={resolved} is below the minimum {minimum} "
            "required by maximum_candidate_probability"
        )
    return resolved, dt_s / resolved, frequency, limit


def recommend_main_timestep(
    plasma: PlasmaSI,
    collision_model: ArgonNullCollisionModel | None = None,
    *,
    collision_subcycling: CollisionSubcyclingConfig | None = None,
) -> TimeStepRecommendation:
    """Return the main-step limits from ion plasma and neutral collisions."""

    omega_pi = math.sqrt(
        plasma.density_m3 * ELEMENTARY_CHARGE_C**2
        / (EPSILON_0_F_M * plasma.ion_mass_kg)
    )
    plasma_limit = 0.1 / omega_pi
    probability = (
        collision_subcycling.maximum_candidate_probability
        if collision_subcycling is not None
        else 0.1
    )
    collision_frequency, collision_limit = collision_timestep_limit_s(
        plasma, probability, collision_model
    )
    recommended = (
        plasma_limit
        if collision_subcycling is not None
        else min(
            plasma_limit,
            math.inf if collision_limit is None else collision_limit,
        )
    )
    collision_steps, collision_dt, _, _ = resolve_collision_substeps(
        plasma,
        recommended,
        collision_subcycling,
        collision_model,
    )
    return TimeStepRecommendation(
        ion_plasma_frequency_s=omega_pi,
        collision_frequency_s=collision_frequency,
        plasma_limit_s=plasma_limit,
        collision_limit_s=collision_limit,
        recommended_dt_s=recommended,
        collision_substeps=collision_steps,
        collision_dt_s=collision_dt,
    )


__all__ = [
    "TimeStepRecommendation",
    "collision_timestep_limit_s",
    "recommend_main_timestep",
    "resolve_collision_substeps",
]
