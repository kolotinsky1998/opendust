"""Public SI-only configuration and result types for OpenDustJAX."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, NamedTuple
import math
import warnings

import jax


ELEMENTARY_CHARGE_C = 1.602176634e-19
EPSILON_0_F_M = 8.8541878128e-12
ELECTRON_MASS_KG = 9.1093837015e-31
BOLTZMANN_J_K = 1.380649e-23
COULOMB_CONSTANT = 1.0 / (4.0 * math.pi * EPSILON_0_F_M)

ChargeMode = Literal["fixed", "oml", "self_consistent"]
ForceBackend = Literal["cuda", "jax"]


def _positive(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class MaxwellianPlasmaSI:
    ion_temperature_K: float
    electron_temperature_K: float
    density_m3: float
    mach: float
    ion_mass_kg: float

    def __post_init__(self) -> None:
        _positive("ion_temperature_K", self.ion_temperature_K)
        _positive("electron_temperature_K", self.electron_temperature_K)
        _positive("density_m3", self.density_m3)
        _positive("ion_mass_kg", self.ion_mass_kg)
        if not math.isfinite(self.mach):
            raise ValueError("mach must be finite")

    @property
    def screening_length_m(self) -> float:
        return math.sqrt(
            EPSILON_0_F_M
            * BOLTZMANN_J_K
            * self.electron_temperature_K
            / (ELEMENTARY_CHARGE_C**2 * self.density_m3)
        )

    @property
    def ion_thermal_speed_m_s(self) -> float:
        return math.sqrt(BOLTZMANN_J_K * self.ion_temperature_K / self.ion_mass_kg)

    @property
    def flow_speed_m_s(self) -> float:
        bohm = math.sqrt(BOLTZMANN_J_K * self.electron_temperature_K / self.ion_mass_kg)
        return self.mach * bohm


@dataclass(frozen=True)
class FieldDrivenPlasmaSI:
    neutral_temperature_K: float
    electron_temperature_K: float
    density_m3: float
    electric_field_V_m: float
    collision_frequency_s: float
    ion_mass_kg: float

    def __post_init__(self) -> None:
        _positive("neutral_temperature_K", self.neutral_temperature_K)
        _positive("electron_temperature_K", self.electron_temperature_K)
        _positive("density_m3", self.density_m3)
        _positive("collision_frequency_s", self.collision_frequency_s)
        _positive("ion_mass_kg", self.ion_mass_kg)
        if not math.isfinite(self.electric_field_V_m):
            raise ValueError("electric_field_V_m must be finite")

    @property
    def screening_length_m(self) -> float:
        return math.sqrt(
            EPSILON_0_F_M
            * BOLTZMANN_J_K
            * self.electron_temperature_K
            / (ELEMENTARY_CHARGE_C**2 * self.density_m3)
        )

    @property
    def ion_thermal_speed_m_s(self) -> float:
        return math.sqrt(BOLTZMANN_J_K * self.neutral_temperature_K / self.ion_mass_kg)

    @property
    def flow_speed_m_s(self) -> float:
        return (
            ELEMENTARY_CHARGE_C
            * self.electric_field_V_m
            / (self.ion_mass_kg * self.collision_frequency_s)
        )


PlasmaSI = MaxwellianPlasmaSI | FieldDrivenPlasmaSI


@dataclass(frozen=True)
class SimulationConfigSI:
    radius_m: float
    height_m: float
    particle_count: int
    steps: int
    dt_s: float
    seed: int = 0
    force_backend: ForceBackend = "cuda"
    step_chunk_size: int = 100
    softening_m: float | None = None

    def __post_init__(self) -> None:
        _positive("radius_m", self.radius_m)
        _positive("height_m", self.height_m)
        _positive("dt_s", self.dt_s)
        if self.particle_count <= 0:
            raise ValueError("particle_count must be positive")
        if self.steps < 0:
            raise ValueError("steps must be non-negative")
        if self.step_chunk_size <= 0:
            raise ValueError("step_chunk_size must be positive")
        if self.force_backend not in ("cuda", "jax"):
            raise ValueError("force_backend must be 'cuda' or 'jax'")
        if self.softening_m is not None and self.softening_m < 0.0:
            raise ValueError("softening_m must be non-negative")


@dataclass(frozen=True)
class DustParticleSI:
    position_m: tuple[float, float, float]
    radius_m: float
    initial_charge_C: float = 0.0
    charge_mode: ChargeMode = "fixed"

    def __post_init__(self) -> None:
        if len(self.position_m) != 3 or not all(math.isfinite(x) for x in self.position_m):
            raise ValueError("position_m must contain three finite values")
        _positive("radius_m", self.radius_m)
        if not math.isfinite(self.initial_charge_C):
            raise ValueError("initial_charge_C must be finite")
        if self.charge_mode not in ("fixed", "oml", "self_consistent"):
            raise ValueError("unknown charge_mode")


@dataclass(frozen=True)
class PolynomialBackgroundConfig:
    enabled: bool = True
    seed: int = 0
    sample_count: int = 65_536
    radial_points_per_screening_length: float = 10.0
    axial_points_per_screening_length: float = 5.0
    source_block_size: int = 4096
    cache_dir: Path | str | None = None

    def __post_init__(self) -> None:
        if self.sample_count < 256:
            raise ValueError("sample_count must be at least 256")
        _positive("radial_points_per_screening_length", self.radial_points_per_screening_length)
        _positive("axial_points_per_screening_length", self.axial_points_per_screening_length)
        if self.source_block_size <= 0:
            raise ValueError("source_block_size must be positive")


@dataclass(frozen=True)
class PlaneDiagnosticConfig:
    nx: int
    nz: int
    slice_thickness_m: float
    start_step: int
    interval: int
    output_prefix: Path | str
    source_block_size: int = 4096

    def __post_init__(self) -> None:
        if self.nx < 2 or self.nz < 2:
            raise ValueError("diagnostic grid dimensions must be at least 2")
        _positive("slice_thickness_m", self.slice_thickness_m)
        if self.start_step < 0:
            raise ValueError("start_step must be non-negative")
        if self.interval <= 0 or self.source_block_size <= 0:
            raise ValueError("interval and source_block_size must be positive")


@dataclass(frozen=True)
class OutputConfig:
    trajectory_interval: int | None = None
    checkpoint_interval: int | None = None
    checkpoint_path: Path | str | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("trajectory_interval", self.trajectory_interval),
            ("checkpoint_interval", self.checkpoint_interval),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when specified")
        if self.checkpoint_interval is not None and self.checkpoint_path is None:
            raise ValueError("checkpoint_path is required with checkpoint_interval")


class SimulationState(NamedTuple):
    positions_m: jax.Array
    velocities_m_s: jax.Array
    forces_N: jax.Array
    dust_charges_C: jax.Array
    step: jax.Array
    prng_key: jax.Array


@dataclass
class SimulationResult:
    final_state: SimulationState
    times_s: object
    dust_charges_C: object
    dust_orbit_forces_N: object
    dust_collection_forces_N: object
    dust_dust_forces_N: object
    dust_external_forces_N: object
    coincident_pair_count: int = 0
    diagnostic_files: tuple[Path, ...] = field(default_factory=tuple)
    trajectory_steps: object | None = None
    trajectory_positions_m: object | None = None


def validate_system(
    plasma: PlasmaSI,
    simulation: SimulationConfigSI,
    dust_particles: tuple[DustParticleSI, ...],
    diagnostics: PlaneDiagnosticConfig | None,
) -> None:
    if isinstance(plasma, FieldDrivenPlasmaSI):
        probability = plasma.collision_frequency_s * simulation.dt_s
        if probability > 1.0:
            raise ValueError("collision_frequency_s * dt_s must not exceed 1")
        if probability > 0.2:
            warnings.warn(
                "collision_frequency_s * dt_s exceeds 0.2; the legacy one-event "
                "Bernoulli collision approximation may be inaccurate",
                RuntimeWarning,
                stacklevel=2,
            )

    radius = simulation.radius_m
    half_height = 0.5 * simulation.height_m
    for index, dust in enumerate(dust_particles):
        x, y, z = dust.position_m
        if math.hypot(x, y) + dust.radius_m > radius or abs(z) + dust.radius_m > half_height:
            raise ValueError(f"dust particle {index} is not fully inside the cylinder")
        if dust.charge_mode == "oml" and not isinstance(plasma, MaxwellianPlasmaSI):
            raise ValueError("OML charging is supported only for Maxwellian plasma")
        if dust.charge_mode == "oml" and len(dust_particles) != 1:
            raise ValueError("OML charging is supported only for an isolated dust particle")
    for left in range(len(dust_particles)):
        for right in range(left + 1, len(dust_particles)):
            a = dust_particles[left]
            b = dust_particles[right]
            separation = math.dist(a.position_m, b.position_m)
            if separation <= a.radius_m + b.radius_m:
                raise ValueError(f"dust particles {left} and {right} overlap")

    if diagnostics is not None:
        if diagnostics.start_step > simulation.steps:
            raise ValueError("diagnostic start_step exceeds the number of steps")
        if diagnostics.slice_thickness_m > 2.0 * radius:
            warnings.warn(
                "diagnostic slice is thicker than the cylinder diameter",
                RuntimeWarning,
                stacklevel=2,
            )
