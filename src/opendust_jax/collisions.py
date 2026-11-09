"""Replaceable ion-neutral collision models."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
from importlib import resources
import math
from pathlib import Path
import sys
from typing import NamedTuple, Protocol

import jax
import jax.numpy as jnp
import numpy as np

from opendust_jax.models import (
    BOLTZMANN_J_K,
    ELEMENTARY_CHARGE_C,
    ArgonPresheathPlasmaSI,
)


_ELASTIC_FILE = "Arplus_Ar_elastic_isotropic_scattering.csv"
_CX_FILE = "Arplus_Ar_resonant_charge_exchange.csv"
_EXPECTED_SHA256 = {
    _ELASTIC_FILE: "64845aed697b4c5c66dcb4cfd6887b77d66f6abc078cdda1492585be93f670e6",
    _CX_FILE: "e095b982e08694dda3abcd80127e9e905f2e70b55e064f0c19b6757fdd3a9173",
}
_REQUIRED_COLUMNS = {
    "source", "process_type", "reaction", "energy_frame", "energy_eV",
    "cross_section_m2", "reference",
}


class CollisionModel(Protocol):
    def apply(
        self,
        key: jax.Array,
        velocities_m_s: jax.Array,
        thermal_speed_m_s: float,
        probability: float,
    ) -> tuple[jax.Array, jax.Array]: ...


class CollisionEvents(NamedTuple):
    candidates: jax.Array
    elastic: jax.Array
    charge_exchange: jax.Array
    null: jax.Array
    out_of_range: jax.Array
    majorant_exceeded: jax.Array


@dataclass(frozen=True)
class CrossSectionData:
    energy_eV: np.ndarray
    elastic_m2: np.ndarray
    charge_exchange_m2: np.ndarray
    checksums: tuple[str, str]
    energy_frame: str

    @property
    def maximum_energy_eV(self) -> float:
        return float(self.energy_eV[-1])


def _data_directory() -> Path:
    source_tree = Path(__file__).resolve().parents[2] / "data" / "cross_sections"
    if source_tree.is_dir():
        return source_tree
    packaged = resources.files("opendust_jax").joinpath("data", "cross_sections")
    if packaged.is_dir():
        return Path(str(packaged))
    installed_data = Path(sys.prefix) / "opendust_jax" / "data" / "cross_sections"
    if installed_data.is_dir():
        return installed_data
    raise FileNotFoundError("packaged Ar+/Ar cross-section data are unavailable")


def _read_table(path: Path, expected_process: str) -> tuple[np.ndarray, np.ndarray, str, str]:
    raw = path.read_bytes()
    checksum = hashlib.sha256(raw).hexdigest()
    expected_checksum = _EXPECTED_SHA256.get(path.name)
    if expected_checksum is not None and checksum != expected_checksum:
        raise ValueError(f"cross-section checksum mismatch for {path.name}")
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or set(reader.fieldnames) != _REQUIRED_COLUMNS:
            raise ValueError(f"unexpected cross-section schema in {path.name}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"empty cross-section table: {path.name}")
    if {row["process_type"] for row in rows} != {expected_process}:
        raise ValueError(f"unexpected process_type in {path.name}")
    frames = {row["energy_frame"] for row in rows}
    if len(frames) != 1:
        raise ValueError(f"mixed energy frames in {path.name}")
    energy = np.asarray([float(row["energy_eV"]) for row in rows], dtype=np.float64)
    sigma = np.asarray([float(row["cross_section_m2"]) for row in rows], dtype=np.float64)
    if not np.all(np.isfinite(energy)) or not np.all(np.diff(energy) > 0.0):
        raise ValueError(f"energies must be finite and strictly increasing in {path.name}")
    if energy[0] != 0.0 or np.any(~np.isfinite(sigma)) or np.any(sigma < 0.0):
        raise ValueError(f"invalid cross sections in {path.name}")
    return energy, sigma, frames.pop(), checksum


def load_argon_cross_sections(directory: Path | str | None = None) -> CrossSectionData:
    root = _data_directory() if directory is None else Path(directory)
    e_el, elastic, frame_el, hash_el = _read_table(
        root / _ELASTIC_FILE, "ion_elastic_isotropic"
    )
    e_cx, charge_exchange, frame_cx, hash_cx = _read_table(
        root / _CX_FILE, "ion_resonant_charge_exchange"
    )
    if frame_el != "center_of_mass" or frame_cx != frame_el:
        raise ValueError("Ar+/Ar cross sections must share the center_of_mass frame")
    if not np.array_equal(e_el, e_cx):
        raise ValueError("Ar+/Ar collision tables must use the same energy grid")
    return CrossSectionData(e_el, elastic, charge_exchange, (hash_el, hash_cx), frame_el)


def _log_interpolate_jax(energy_eV: jax.Array, grid: jax.Array, values: jax.Array) -> jax.Array:
    safe_energy = jnp.maximum(energy_eV, grid[1])
    interpolated = jnp.exp(
        jnp.interp(jnp.log(safe_energy), jnp.log(grid[1:]), jnp.log(values[1:]))
    )
    return jnp.where(energy_eV <= grid[0], values[0], interpolated)


def elastic_scatter_equal_mass(
    ion_velocity_m_s: jax.Array,
    neutral_velocity_m_s: jax.Array,
    direction: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Return the two post-collision velocities for isotropic equal-mass scattering."""
    relative_speed = jnp.linalg.norm(ion_velocity_m_s - neutral_velocity_m_s, axis=-1)
    unit = direction / jnp.maximum(
        jnp.linalg.norm(direction, axis=-1, keepdims=True), 1.0e-20
    )
    center = 0.5 * (ion_velocity_m_s + neutral_velocity_m_s)
    half_relative = 0.5 * relative_speed[..., None] * unit
    return center + half_relative, center - half_relative


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


class ArgonNullCollisionModel:
    """Phelps Ar+/Ar elastic and resonant-CX null-collision operator."""

    def __init__(
        self,
        plasma: ArgonPresheathPlasmaSI,
        data: CrossSectionData | None = None,
        *,
        maximum_candidate_probability: float = 0.1,
        maximum_relative_speed_m_s: float | None = None,
    ) -> None:
        self.plasma = plasma
        self.data = data or load_argon_cross_sections()
        self.maximum_candidate_probability = maximum_candidate_probability
        if not 0.0 < maximum_candidate_probability < 1.0:
            raise ValueError("maximum_candidate_probability must lie between zero and one")
        if maximum_relative_speed_m_s is not None and (
            not math.isfinite(maximum_relative_speed_m_s)
            or maximum_relative_speed_m_s <= 0.0
        ):
            raise ValueError("maximum_relative_speed_m_s must be finite and positive")
        self.maximum_relative_speed_m_s = maximum_relative_speed_m_s
        self._energy = jnp.asarray(self.data.energy_eV, dtype=jnp.float32)
        self._elastic = jnp.asarray(self.data.elastic_m2, dtype=jnp.float32)
        self._cx = jnp.asarray(self.data.charge_exchange_m2, dtype=jnp.float32)
        reduced_mass = 0.5 * plasma.ion_mass_kg
        table_maximum = self.data.energy_eV[-1]
        requested_maximum = (
            table_maximum
            if maximum_relative_speed_m_s is None
            else 0.5
            * reduced_mass
            * maximum_relative_speed_m_s**2
            / ELEMENTARY_CHARGE_C
        )
        self.maximum_majorant_energy_eV = float(min(requested_maximum, table_maximum))
        dense_energy = np.geomspace(
            max(self.data.energy_eV[1], 1.0e-12),
            self.maximum_majorant_energy_eV,
            200_000,
        )
        elastic = np.exp(np.interp(
            np.log(dense_energy), np.log(self.data.energy_eV[1:]), np.log(self.data.elastic_m2[1:])
        ))
        cx = np.exp(np.interp(
            np.log(dense_energy), np.log(self.data.energy_eV[1:]), np.log(self.data.charge_exchange_m2[1:])
        ))
        relative_speed = np.sqrt(2.0 * dense_energy * ELEMENTARY_CHARGE_C / reduced_mass)
        self.maximum_rate_coefficient_m3_s = float(np.max((elastic + cx) * relative_speed) * 1.0001)
        self.maximum_frequency_s = plasma.neutral_density_m3 * self.maximum_rate_coefficient_m3_s

    @property
    def checksums(self) -> tuple[str, str]:
        return self.data.checksums

    def validate_timestep(self, dt_s: float) -> None:
        probability = -math.expm1(-self.maximum_frequency_s * dt_s)
        if probability > self.maximum_candidate_probability:
            raise ValueError(
                "null-collision candidate probability exceeds "
                f"{self.maximum_candidate_probability:g}; reduce dt_s"
            )

    def cross_sections_m2(self, energy_eV: jax.Array) -> tuple[jax.Array, jax.Array]:
        return (
            _log_interpolate_jax(energy_eV, self._energy, self._elastic),
            _log_interpolate_jax(energy_eV, self._energy, self._cx),
        )

    def apply(
        self,
        key: jax.Array,
        velocities_m_s: jax.Array,
        thermal_speed_m_s: float,
        dt_s: float,
    ) -> tuple[jax.Array, CollisionEvents]:
        candidate_key, neutral_key, accept_key, channel_key, direction_key = jax.random.split(key, 5)
        count = velocities_m_s.shape[0]
        candidate_probability = -jnp.expm1(-self.maximum_frequency_s * dt_s)
        candidates = jax.random.bernoulli(candidate_key, candidate_probability, (count,))
        neutral = thermal_speed_m_s * jax.random.normal(
            neutral_key, velocities_m_s.shape, dtype=jnp.float32
        )
        relative = velocities_m_s - neutral
        relative_speed = jnp.linalg.norm(relative, axis=1)
        reduced_mass = 0.5 * self.plasma.ion_mass_kg
        energy_eV = 0.5 * reduced_mass * relative_speed**2 / ELEMENTARY_CHARGE_C
        out_of_range = candidates & (energy_eV > self.data.maximum_energy_eV)
        bounded_energy = jnp.minimum(energy_eV, self.data.maximum_energy_eV)
        elastic_sigma, cx_sigma = self.cross_sections_m2(bounded_energy)
        elastic_rate = self.plasma.neutral_density_m3 * elastic_sigma * relative_speed
        cx_rate = self.plasma.neutral_density_m3 * cx_sigma * relative_speed
        total_rate = elastic_rate + cx_rate
        majorant_exceeded = total_rate > self.maximum_frequency_s
        effective_rate = jnp.minimum(total_rate, self.maximum_frequency_s)
        real = candidates & (
            jax.random.uniform(accept_key, (count,), dtype=jnp.float32)
            * self.maximum_frequency_s
            < effective_rate
        )
        elastic = real & (
            jax.random.uniform(channel_key, (count,), dtype=jnp.float32) * total_rate
            < elastic_rate
        )
        charge_exchange = real & ~elastic
        null = candidates & ~real

        direction = jax.random.normal(direction_key, velocities_m_s.shape, dtype=jnp.float32)
        scattered, _ = elastic_scatter_equal_mass(velocities_m_s, neutral, direction)
        updated = jnp.where(elastic[:, None], scattered, velocities_m_s)
        updated = jnp.where(charge_exchange[:, None], neutral, updated)
        return updated, CollisionEvents(
            candidates,
            elastic,
            charge_exchange,
            null,
            out_of_range,
            majorant_exceeded,
        )


def presheath_preparation_speed_limit_m_s(
    plasma: ArgonPresheathPlasmaSI,
) -> float:
    """Return the fixed 2 cs + 6 vth preparation majorant speed."""
    sound_speed = math.sqrt(
        BOLTZMANN_J_K * plasma.electron_temperature_K / plasma.ion_mass_kg
    )
    return 2.0 * sound_speed + 6.0 * plasma.ion_thermal_speed_m_s


__all__ = [
    "ArgonNullCollisionModel",
    "CollisionEvents",
    "CollisionModel",
    "CrossSectionData",
    "LegacyChargeExchangeCollision",
    "elastic_scatter_equal_mass",
    "load_argon_cross_sections",
    "presheath_preparation_speed_limit_m_s",
]
