"""Replaceable ion-neutral collision models."""

from __future__ import annotations

from typing import Protocol

import jax
import jax.numpy as jnp


class CollisionModel(Protocol):
    def apply(
        self,
        key: jax.Array,
        velocities_m_s: jax.Array,
        thermal_speed_m_s: float,
        probability: float,
    ) -> tuple[jax.Array, jax.Array]: ...


class LegacyChargeExchangeCollision:
    """One Bernoulli event per ion, replacing the complete velocity vector."""

    def apply(
        self,
        key: jax.Array,
        velocities_m_s: jax.Array,
        thermal_speed_m_s: float,
        probability: float,
    ) -> tuple[jax.Array, jax.Array]:
        event_key, velocity_key = jax.random.split(key)
        events = jax.random.bernoulli(
            event_key, probability, (velocities_m_s.shape[0],)
        )
        neutral = thermal_speed_m_s * jax.random.normal(
            velocity_key, velocities_m_s.shape, dtype=jnp.float32
        )
        return jnp.where(events[:, None], neutral, velocities_m_s), events


__all__ = ["CollisionModel", "LegacyChargeExchangeCollision"]
