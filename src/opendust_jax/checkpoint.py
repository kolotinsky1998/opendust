"""Restart files for OpenDustJAX."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from opendust_jax.models import SimulationState


def save_checkpoint(
    path: Path | str,
    state: SimulationState,
    background_coefficients: np.ndarray | jax.Array,
    metadata: dict | None = None,
) -> Path:
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    host = jax.device_get(state)
    with destination.open("wb") as stream:
        np.savez_compressed(
            stream,
            positions_m=np.asarray(host.positions_m, dtype=np.float32),
            velocities_m_s=np.asarray(host.velocities_m_s, dtype=np.float32),
            forces_N=np.asarray(host.forces_N, dtype=np.float32),
            dust_charges_C=np.asarray(host.dust_charges_C, dtype=np.float32),
            step=np.asarray(host.step, dtype=np.int64),
            prng_key=np.asarray(host.prng_key, dtype=np.uint32),
            background_coefficients=np.asarray(background_coefficients, dtype=np.float64),
            metadata_json=np.asarray(json.dumps(metadata or {}, sort_keys=True)),
        )
    return destination


def load_checkpoint(path: Path | str) -> tuple[SimulationState, np.ndarray, dict]:
    source = Path(path).expanduser()
    with np.load(source, allow_pickle=False) as data:
        state = SimulationState(
            positions_m=jnp.asarray(data["positions_m"], dtype=jnp.float32),
            velocities_m_s=jnp.asarray(data["velocities_m_s"], dtype=jnp.float32),
            forces_N=jnp.asarray(data["forces_N"], dtype=jnp.float32),
            dust_charges_C=jnp.asarray(data["dust_charges_C"], dtype=jnp.float32),
            step=jnp.asarray(data["step"], dtype=jnp.int32),
            prng_key=jnp.asarray(data["prng_key"], dtype=jnp.uint32),
        )
        coefficients = np.asarray(data["background_coefficients"], dtype=np.float64)
        metadata = json.loads(str(data["metadata_json"]))
    return state, coefficients, metadata
