"""Single-NVIDIA-GPU OpenDust simulation engine using JAX and CUDA FFI."""

from __future__ import annotations

from dataclasses import asdict
from fractions import Fraction
import math
from pathlib import Path
from typing import NamedTuple
import warnings

import jax
import jax.numpy as jnp
import numpy as np
from scipy import special

from opendust_jax.background import PolynomialBackground, build_polynomial_background
from opendust_jax.checkpoint import load_checkpoint, save_checkpoint
from opendust_jax.collisions import (
    ArgonNullCollisionModel,
    CollisionEvents,
    CollisionModel,
    LegacyChargeExchangeCollision,
)
from opendust_jax.cuda_pair_forces import (
    cuda_pair_forces_available,
    dust_orbit_forces_cuda,
    ion_dust_forces_cuda,
    softened_yukawa_forces_cuda_with_stats,
)
from opendust_jax.diagnostics import (
    diagnostic_axes,
    plane_cic_density_snapshot_m3,
    plane_potential_snapshot_V,
    write_plane_diagnostics,
)
from opendust_jax.external_forces import (
    dust_orbit_forces_jax,
    ion_dust_forces_jax,
)
from opendust_jax.models import (
    ArgonPresheathPlasmaSI,
    BOLTZMANN_J_K,
    COULOMB_CONSTANT,
    CollisionStatistics,
    ELEMENTARY_CHARGE_C,
    DustParticleSI,
    FieldDrivenPlasmaSI,
    MaxwellianPlasmaSI,
    OutputConfig,
    PlaneDiagnosticConfig,
    PlasmaSI,
    PolynomialBackgroundConfig,
    SimulationConfigSI,
    SimulationResult,
    SimulationState,
    validate_system,
)
from opendust_jax.presheath import CylindricalSplineBackground
from opendust_jax.pair_forces import softened_yukawa_forces
from opendust_jax.physics import (
    background_force_N,
    dust_dust_forces_N,
    update_dust_charges_C,
)
from opendust_jax.timestepping import resolve_collision_substeps


class _RunCarry(NamedTuple):
    state: SimulationState
    pair_forces_N: jax.Array
    background_forces_N: jax.Array
    fast_forces_N: jax.Array
    potential_sum: jax.Array
    density_sum: jax.Array
    sample_count: jax.Array
    coincident_count: jax.Array
    collision_candidates: jax.Array
    collision_elastic: jax.Array
    collision_charge_exchange: jax.Array
    collision_null: jax.Array
    collision_out_of_range: jax.Array
    collision_majorant_exceeded: jax.Array
    collision_operator_calls: jax.Array


def _merged_substep_schedule(
    dt_s: float, collision_substeps: int, charging_substeps: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return segment durations and collision indices on the union time grid."""
    boundaries = {
        *(Fraction(index, collision_substeps) for index in range(collision_substeps + 1)),
        *(Fraction(index, charging_substeps) for index in range(charging_substeps + 1)),
    }
    ordered = sorted(boundaries)
    durations = np.asarray(
        [float(right - left) * dt_s for left, right in zip(ordered[:-1], ordered[1:])],
        dtype=np.float32,
    )
    collision_indices = []
    for left in ordered[:-1]:
        scaled = left * collision_substeps
        collision_indices.append(
            int(scaled) if scaled.denominator == 1 and int(scaled) < collision_substeps else -1
        )
    return durations, np.asarray(collision_indices, dtype=np.int32)


def _normal_pdf(values: np.ndarray, mean: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((values - mean) / sigma) ** 2) / (
        math.sqrt(2.0 * math.pi) * sigma
    )


def _axial_pdf(
    values: np.ndarray, plasma: PlasmaSI, thermal_speed: float, flow_speed: float
) -> np.ndarray:
    if isinstance(plasma, (MaxwellianPlasmaSI, ArgonPresheathPlasmaSI)) or abs(flow_speed) < 1.0e-12 * thermal_speed:
        return _normal_pdf(values, flow_speed, thermal_speed)
    sign = 1.0 if flow_speed >= 0.0 else -1.0
    reflected = sign * values
    tau = abs(flow_speed)
    argument = (thermal_speed**2 / tau - reflected) / (
        math.sqrt(2.0) * thermal_speed
    )
    exponent = thermal_speed**2 / (2.0 * tau**2) - reflected / tau
    # erfcx prevents overflow in the far negative tail.
    pdf = np.exp(-0.5 * (reflected / thermal_speed) ** 2) * special.erfcx(argument) / (2.0 * tau)
    pdf = np.where(np.isfinite(pdf), pdf, np.exp(np.clip(exponent, -745, 700)) * special.erfc(argument) / (2.0 * tau))
    return np.maximum(pdf, 0.0)


def _cdf(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    increments = 0.5 * (weights[1:] + weights[:-1]) * np.diff(values)
    result = np.concatenate(([0.0], np.cumsum(increments)))
    if result[-1] <= 0.0 or not np.isfinite(result[-1]):
        raise ValueError("velocity distribution has zero or invalid normalization")
    result /= result[-1]
    result[-1] = 1.0
    return np.maximum.accumulate(result)


def _velocity_tables(plasma: PlasmaSI) -> dict[str, np.ndarray | float]:
    v_t = plasma.ion_thermal_speed_m_s
    flow = plasma.flow_speed_m_s
    extent = max(10.0 * v_t, abs(flow) + 12.0 * v_t, 12.0 * abs(flow))
    values = np.linspace(-extent, extent, 8193, dtype=np.float64)
    pdf = _axial_pdf(values, plasma, v_t, flow)
    base_cdf = _cdf(values, pdf)
    positive = values >= 0.0
    negative = values <= 0.0
    bottom_values = values[positive]
    top_values = values[negative]
    bottom_weights = bottom_values * pdf[positive]
    top_weights = -top_values * pdf[negative]
    bottom_cdf = _cdf(bottom_values, bottom_weights)
    top_cdf = _cdf(top_values, top_weights)
    bottom_flux = float(
        np.sum(0.5 * (bottom_weights[1:] + bottom_weights[:-1]) * np.diff(bottom_values))
    )
    top_flux = float(
        np.sum(0.5 * (top_weights[1:] + top_weights[:-1]) * np.diff(top_values))
    )
    side_flux_per_area = v_t / math.sqrt(2.0 * math.pi)
    return {
        "values": values.astype(np.float32),
        "base_cdf": base_cdf.astype(np.float32),
        "bottom_values": bottom_values.astype(np.float32),
        "bottom_cdf": bottom_cdf.astype(np.float32),
        "top_values": top_values.astype(np.float32),
        "top_cdf": top_cdf.astype(np.float32),
        "bottom_flux": bottom_flux,
        "top_flux": top_flux,
        "side_flux_per_area": side_flux_per_area,
    }


class OpenDustJAX:
    """SI-only OpenDust implementation for one NVIDIA GPU.

    ``force_backend='jax'`` is an intentionally slow reference path used by
    unit tests. Production simulations are required to use ``'cuda'``.
    """

    def __init__(
        self,
        plasma: PlasmaSI,
        simulation: SimulationConfigSI,
        dust_particles: list[DustParticleSI] | tuple[DustParticleSI, ...] = (),
        *,
        background: PolynomialBackgroundConfig | CylindricalSplineBackground | None = None,
        diagnostics: PlaneDiagnosticConfig | None = None,
        output: OutputConfig | None = None,
        collision_model: CollisionModel | None = None,
    ) -> None:
        self.plasma = plasma
        self.simulation = simulation
        self.dust_particles = tuple(dust_particles)
        self.background_config = (
            background if isinstance(background, PolynomialBackgroundConfig) else None
        )
        self.diagnostics = diagnostics
        self.output = output or OutputConfig()
        if isinstance(plasma, ArgonPresheathPlasmaSI):
            if not isinstance(background, CylindricalSplineBackground):
                raise ValueError(
                    "ArgonPresheathPlasmaSI requires an explicitly prepared "
                    "CylindricalSplineBackground"
                )
            self.collision_model = collision_model or ArgonNullCollisionModel(plasma)
            if not isinstance(self.collision_model, ArgonNullCollisionModel):
                raise TypeError("Argon presheath requires ArgonNullCollisionModel")
            if simulation.collision_subcycling is None:
                self.collision_model.validate_timestep(simulation.dt_s)
            self.background = background
        else:
            if isinstance(background, CylindricalSplineBackground):
                raise TypeError("cylindrical backgrounds require ArgonPresheathPlasmaSI")
            self.collision_model = collision_model or LegacyChargeExchangeCollision()
        validate_system(plasma, simulation, self.dust_particles, diagnostics)
        if simulation.force_backend == "cuda" and not cuda_pair_forces_available():
            raise RuntimeError(
                "The OpenDustJAX CUDA FFI library is unavailable. Build it with "
                "scripts/build_cuda_pair_forces.py on an NVIDIA GPU host."
            )
        if diagnostics is not None and not jax.config.x64_enabled:
            warnings.warn(
                "JAX x64 is disabled; diagnostic temporal sums will use FP32. "
                "Set JAX_ENABLE_X64=1 for FP64 accumulation.",
                RuntimeWarning,
                stacklevel=2,
            )

        self.screening_length_m = plasma.screening_length_m
        self.softening_m = (
            simulation.softening_m
            if simulation.softening_m is not None
            else 1.0e-3 * self.screening_length_m
        )
        volume = math.pi * simulation.radius_m**2 * simulation.height_m
        self.superion_weight = plasma.density_m3 * volume / simulation.particle_count
        self.superion_charge_C = ELEMENTARY_CHARGE_C * self.superion_weight
        self.superion_mass_kg = plasma.ion_mass_kg * self.superion_weight
        if not isinstance(plasma, ArgonPresheathPlasmaSI):
            self.background_config = self.background_config or PolynomialBackgroundConfig()
            self.background = build_polynomial_background(
                simulation.radius_m,
                simulation.height_m,
                self.screening_length_m,
                simulation.particle_count,
                self.background_config,
            )
        self._velocity = _velocity_tables(plasma)
        self._dust_positions = jnp.asarray(
            [d.position_m for d in self.dust_particles], dtype=jnp.float32
        ).reshape((-1, 3))
        self._dust_radii = jnp.asarray(
            [d.radius_m for d in self.dust_particles], dtype=jnp.float32
        )
        mode_values = {"fixed": 0, "oml": 1, "self_consistent": 2}
        self._dust_modes = jnp.asarray(
            [mode_values[d.charge_mode] for d in self.dust_particles], dtype=jnp.int32
        )
        self.charging_substeps = self._resolve_charging_substeps()
        self.charging_dt_s = simulation.dt_s / self.charging_substeps
        (
            self.collision_substeps,
            self.collision_dt_s,
            _,
            _,
        ) = resolve_collision_substeps(
            plasma,
            simulation.dt_s,
            simulation.collision_subcycling,
            self.collision_model
            if isinstance(self.collision_model, ArgonNullCollisionModel)
            else None,
        )
        if isinstance(plasma, ArgonPresheathPlasmaSI):
            self.collision_model.validate_timestep(self.collision_dt_s)
            self._validate_presheath_background()
        durations, collision_indices = _merged_substep_schedule(
            simulation.dt_s, self.collision_substeps, self.charging_substeps
        )
        self.integration_segments = len(durations)
        self.maximum_fast_dt_s = float(np.max(durations))
        self._segment_durations_host = durations
        self._segment_durations_s = jnp.asarray(durations, dtype=jnp.float32)
        self._segment_collision_indices = jnp.asarray(
            collision_indices, dtype=jnp.int32
        )
        active_collision_indices = np.maximum.accumulate(collision_indices)
        self._segment_active_collision_indices = jnp.asarray(
            active_collision_indices, dtype=jnp.int32
        )
        self._segment_indices = jnp.arange(self.integration_segments, dtype=jnp.int32)
        self._background_coefficients = jnp.asarray(
            (
                np.zeros(15)
                if isinstance(self.background, CylindricalSplineBackground)
                else self.background.device_coefficients
            ),
            dtype=jnp.float32,
        )
        if diagnostics is not None:
            self._x_axis, self._z_axis = diagnostic_axes(
                simulation.radius_m, simulation.height_m, diagnostics
            )
        else:
            self._x_axis = self._z_axis = jnp.zeros((0,), jnp.float32)
        # Keep compiled scan executables across run() calls.  Besides avoiding
        # needless recompilation in repeated production runs, this makes a
        # benchmark warm-up actually exclude JIT compilation from timing.
        self._compiled_chunks: dict[int, object] = {}

    def _resolve_charging_substeps(self) -> int:
        config = self.simulation.charging_subcycling
        if config is None:
            return 1
        self_consistent_radii = [
            dust.radius_m
            for dust in self.dust_particles
            if dust.charge_mode == "self_consistent"
        ]
        if self_consistent_radii:
            velocity_limit = (
                config.velocity_limit_m_s
                if config.velocity_limit_m_s is not None
                else abs(self.plasma.flow_speed_m_s)
                + config.thermal_speed_multiplier
                * self.plasma.ion_thermal_speed_m_s
            )
            automatic = max(
                1,
                math.ceil(
                    self.simulation.dt_s
                    * velocity_limit
                    / (
                        config.max_displacement_fraction
                        * min(self_consistent_radii)
                    )
                ),
            )
        else:
            automatic = 1
        if config.substeps is None:
            return automatic
        if self_consistent_radii and config.substeps < automatic:
            warnings.warn(
                f"explicit charging substeps={config.substeps} is below the "
                f"automatic displacement estimate {automatic}",
                RuntimeWarning,
                stacklevel=2,
            )
        return config.substeps

    def _validate_presheath_background(self) -> None:
        metadata = self.background.metadata
        saved_plasma = metadata.get("plasma")
        if saved_plasma != asdict(self.plasma):
            raise ValueError("presheath background plasma configuration does not match")
        saved_simulation = metadata.get("simulation", {})
        for name in ("radius_m", "height_m", "particle_count", "seed"):
            if saved_simulation.get(name) != getattr(self.simulation, name):
                raise ValueError(f"presheath background {name} does not match")
        expected_softening = self.simulation.softening_m or 1.0e-3 * self.plasma.screening_length_m
        if not math.isclose(float(metadata.get("softening_m", -1.0)), expected_softening):
            raise ValueError("presheath background softening_m does not match")
        checksums = tuple(metadata.get("cross_section_sha256", ()))
        if checksums != tuple(self.collision_model.checksums):
            raise ValueError("presheath background cross-section data do not match")

    def _axial_electric_field_V_m(self, z_m: jax.Array) -> jax.Array:
        if isinstance(self.plasma, ArgonPresheathPlasmaSI):
            field = self.plasma.electric_field
            lower = -jnp.inf if field.z_min_m is None else field.z_min_m
            upper = jnp.inf if field.z_max_m is None else field.z_max_m
            bounded = jnp.clip(z_m, lower, upper)
            return field.E0_V_m + field.alpha_V_m2 * bounded + field.beta_V_m3 * bounded**2
        if isinstance(self.plasma, FieldDrivenPlasmaSI):
            return jnp.full_like(z_m, self.plasma.electric_field_V_m)
        return jnp.zeros_like(z_m)

    def _checkpoint_background_array(self) -> np.ndarray:
        if isinstance(self.background, CylindricalSplineBackground):
            return np.empty((0,), dtype=np.float64)
        return self.background.coefficients

    def _initial_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.simulation.seed)
        count = self.simulation.particle_count
        positions = np.empty((count, 3), dtype=np.float32)
        filled = 0
        attempts = 0
        dust_positions = np.asarray([d.position_m for d in self.dust_particles])
        dust_radii = np.asarray([d.radius_m for d in self.dust_particles])
        while filled < count:
            attempts += 1
            if attempts > 1000:
                raise RuntimeError("could not sample positions outside dust particles")
            batch_count = max(1024, 2 * (count - filled))
            radius = self.simulation.radius_m * np.sqrt(rng.random(batch_count))
            theta = 2.0 * np.pi * rng.random(batch_count)
            batch = np.stack(
                (
                    radius * np.cos(theta),
                    radius * np.sin(theta),
                    self.simulation.height_m * (rng.random(batch_count) - 0.5),
                ),
                axis=1,
            )
            if len(self.dust_particles):
                distance = np.linalg.norm(
                    batch[:, None, :] - dust_positions[None, :, :], axis=-1
                )
                batch = batch[np.all(distance > dust_radii[None, :], axis=1)]
            take = min(len(batch), count - filled)
            positions[filled : filled + take] = batch[:take]
            filled += take

        v_t = self.plasma.ion_thermal_speed_m_s
        velocities = np.empty((count, 3), dtype=np.float32)
        velocities[:, :2] = rng.normal(0.0, v_t, size=(count, 2))
        velocities[:, 2] = np.interp(
            rng.random(count), self._velocity["base_cdf"], self._velocity["values"]
        )
        return positions, velocities

    def _pair_force(self, positions: jax.Array) -> tuple[jax.Array, jax.Array]:
        prefactor = COULOMB_CONSTANT * self.superion_charge_C**2
        if self.simulation.force_backend == "cuda":
            pair, coincident = softened_yukawa_forces_cuda_with_stats(
                positions, prefactor, self.screening_length_m, self.softening_m
            )
        else:
            pair = softened_yukawa_forces(
                positions,
                prefactor,
                self.screening_length_m,
                self.softening_m,
                block_size=min(256, self.simulation.particle_count),
            )
            coincident = jnp.asarray(0, jnp.int32)
        return pair, coincident

    def _fast_force(self, positions: jax.Array, dust_charges: jax.Array) -> jax.Array:
        electric_field = (
            self.plasma.electric_field_V_m
            if isinstance(self.plasma, FieldDrivenPlasmaSI)
            else 0.0
        )
        if self.simulation.force_backend == "cuda":
            external = ion_dust_forces_cuda(
                positions,
                self._dust_positions,
                dust_charges,
                self.superion_charge_C,
                electric_field,
            )
        else:
            external = ion_dust_forces_jax(
                positions,
                self._dust_positions,
                dust_charges,
                self.superion_charge_C,
                electric_field,
            )
        if isinstance(self.background, CylindricalSplineBackground):
            field_z = self._axial_electric_field_V_m(positions[:, 2])
            external = external.at[:, 2].add(self.superion_charge_C * field_z)
        return external

    def _background_force(self, positions: jax.Array) -> jax.Array:
        if isinstance(self.background, CylindricalSplineBackground):
            return self.background.force_N(positions, self.superion_charge_C)
        return background_force_N(
            positions,
            self.superion_charge_C,
            self._background_coefficients,
            self.screening_length_m,
            self.superion_charge_C,
        )

    def _orbit_force(
        self, positions: jax.Array, dust_charges: jax.Array
    ) -> jax.Array:
        if self.simulation.force_backend == "cuda":
            return dust_orbit_forces_cuda(
                positions,
                self._dust_positions,
                dust_charges,
                self.superion_charge_C,
                self.screening_length_m,
            )
        return dust_orbit_forces_jax(
            positions,
            self._dust_positions,
            dust_charges,
            self.superion_charge_C,
            self.screening_length_m,
        )

    def _force(
        self, positions: jax.Array, dust_charges: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        pair, coincident = self._pair_force(positions)
        fast = self._fast_force(positions, dust_charges)
        background = self._background_force(positions)
        orbit = self._orbit_force(positions, dust_charges)
        total = pair + background + fast
        return total, orbit, coincident

    def initial_state(self) -> SimulationState:
        positions, velocities = self._initial_arrays()
        charges = jnp.asarray(
            [d.initial_charge_C for d in self.dust_particles], dtype=jnp.float32
        )
        device_positions = jnp.asarray(positions)
        pair, _ = self._pair_force(device_positions)
        force = (
            pair
            + self._background_force(device_positions)
            + self._fast_force(device_positions, charges)
        )
        return SimulationState(
            device_positions,
            jnp.asarray(velocities),
            force,
            charges,
            jnp.asarray(0, jnp.int32),
            jax.random.PRNGKey(self.simulation.seed),
        )

    def _reinject(
        self, positions: jax.Array, velocities: jax.Array, mask: jax.Array, key: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        if isinstance(self.plasma, ArgonPresheathPlasmaSI):
            return self._reinject_presheath(positions, velocities, mask, key)
        count = positions.shape[0]
        keys = jax.random.split(key, 9)
        uniform = [jax.random.uniform(k, (count,), dtype=jnp.float32) for k in keys[:6]]
        gaussian = jax.random.normal(keys[6], (count, 3), dtype=jnp.float32)
        u_surface, u_r, u_theta, u_z, u_velocity, u_radial = uniform
        radius, half_height = self.simulation.radius_m, 0.5 * self.simulation.height_m
        area_end = math.pi * radius**2
        bottom_flux = area_end * float(self._velocity["bottom_flux"])
        top_flux = area_end * float(self._velocity["top_flux"])
        side_flux = (
            2.0 * math.pi * radius * self.simulation.height_m
            * float(self._velocity["side_flux_per_area"])
        )
        total_flux = bottom_flux + side_flux + top_flux
        p_bottom = bottom_flux / total_flux
        p_side_end = (bottom_flux + side_flux) / total_flux
        bottom = u_surface < p_bottom
        side = (u_surface >= p_bottom) & (u_surface < p_side_end)
        top = ~(bottom | side)

        angle = 2.0 * jnp.pi * u_theta
        disk_radius = radius * jnp.sqrt(u_r)
        end_position = jnp.stack(
            (disk_radius * jnp.cos(angle), disk_radius * jnp.sin(angle),
             jnp.where(bottom, -half_height, half_height)), axis=1
        )
        side_position = jnp.stack(
            (radius * jnp.cos(angle), radius * jnp.sin(angle),
             self.simulation.height_m * (u_z - 0.5)), axis=1
        )
        candidate_position = jnp.where(side[:, None], side_position, end_position)

        bottom_vz = jnp.interp(
            u_velocity,
            jnp.asarray(self._velocity["bottom_cdf"]),
            jnp.asarray(self._velocity["bottom_values"]),
        )
        top_vz = jnp.interp(
            u_velocity,
            jnp.asarray(self._velocity["top_cdf"]),
            jnp.asarray(self._velocity["top_values"]),
        )
        end_velocity = jnp.stack(
            (self.plasma.ion_thermal_speed_m_s * gaussian[:, 0],
             self.plasma.ion_thermal_speed_m_s * gaussian[:, 1],
             jnp.where(bottom, bottom_vz, top_vz)), axis=1
        )
        radial_speed = self.plasma.ion_thermal_speed_m_s * jnp.sqrt(
            -2.0 * jnp.log(jnp.maximum(1.0 - u_radial, 1.0e-7))
        )
        tangential = self.plasma.ion_thermal_speed_m_s * gaussian[:, 1]
        axial = jnp.interp(
            u_velocity,
            jnp.asarray(self._velocity["base_cdf"]),
            jnp.asarray(self._velocity["values"]),
        )
        cos_angle, sin_angle = jnp.cos(angle), jnp.sin(angle)
        side_velocity = jnp.stack(
            (-radial_speed * cos_angle - tangential * sin_angle,
             -radial_speed * sin_angle + tangential * cos_angle, axial), axis=1
        )
        candidate_velocity = jnp.where(side[:, None], side_velocity, end_velocity)
        return (
            jnp.where(mask[:, None], candidate_position, positions),
            jnp.where(mask[:, None], candidate_velocity, velocities),
        )

    def _reinject_presheath(
        self, positions: jax.Array, velocities: jax.Array, mask: jax.Array, key: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        count = positions.shape[0]
        uniform_key, normal_key = jax.random.split(key)
        uniform = jax.random.uniform(uniform_key, (count, 3), dtype=jnp.float32)
        normal = jax.random.normal(normal_key, (count, 2), dtype=jnp.float32)
        radius = self.simulation.radius_m * jnp.sqrt(uniform[:, 0])
        angle = 2.0 * jnp.pi * uniform[:, 1]
        candidate_position = jnp.stack(
            (
                radius * jnp.cos(angle),
                radius * jnp.sin(angle),
                jnp.full((count,), -0.5 * self.simulation.height_m, jnp.float32),
            ),
            axis=1,
        )
        axial = jnp.interp(
            uniform[:, 2],
            jnp.asarray(self._velocity["bottom_cdf"]),
            jnp.asarray(self._velocity["bottom_values"]),
        )
        candidate_velocity = jnp.stack(
            (
                self.plasma.ion_thermal_speed_m_s * normal[:, 0],
                self.plasma.ion_thermal_speed_m_s * normal[:, 1],
                axial,
            ),
            axis=1,
        )
        return (
            jnp.where(mask[:, None], candidate_position, positions),
            jnp.where(mask[:, None], candidate_velocity, velocities),
        )

    def _reflect_presheath_side(
        self, positions: jax.Array, velocities: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        radial = jnp.sqrt(positions[:, 0] ** 2 + positions[:, 1] ** 2)
        outside = radial > self.simulation.radius_m
        unit = positions[:, :2] / jnp.maximum(radial[:, None], 1.0e-20)
        normal_velocity = jnp.sum(velocities[:, :2] * unit, axis=1)
        reflected_velocity = velocities.at[:, :2].add(
            jnp.where(outside[:, None], -2.0 * normal_velocity[:, None] * unit, 0.0)
        )
        reflected_radius = jnp.clip(
            2.0 * self.simulation.radius_m - radial,
            0.0,
            self.simulation.radius_m,
        )
        reflected_positions = positions.at[:, :2].set(
            jnp.where(outside[:, None], reflected_radius[:, None] * unit, positions[:, :2])
        )
        return reflected_positions, reflected_velocity

    def _dust_force_components(
        self, positions: jax.Array, charges: jax.Array, orbit_in: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        if isinstance(self.background, CylindricalSplineBackground):
            orbit = orbit_in + self.background.force_N(self._dust_positions, charges)
        else:
            orbit = orbit_in + background_force_N(
                self._dust_positions,
                charges,
                self._background_coefficients,
                self.screening_length_m,
                self.superion_charge_C,
            )
        dust_dust = dust_dust_forces_N(self._dust_positions, charges)
        electric_field = self._axial_electric_field_V_m(self._dust_positions[:, 2])
        external = jnp.zeros_like(self._dust_positions).at[:, 2].set(charges * electric_field)
        return orbit, dust_dust, external

    def _complete_carry(
        self,
        carry: _RunCarry,
        next_state: SimulationState,
        pair_forces_N: jax.Array,
        coincident: jax.Array,
        collision_events: CollisionEvents,
        background_forces_N: jax.Array | None = None,
        fast_forces_N: jax.Array | None = None,
        collision_counts: tuple[jax.Array, ...] | None = None,
        collision_operator_calls: jax.Array | int = 0,
    ) -> _RunCarry:
        positions = next_state.positions_m
        if self.diagnostics is not None:
            should_sample = (next_state.step >= self.diagnostics.start_step) & (
                (next_state.step - self.diagnostics.start_step)
                % self.diagnostics.interval == 0
            )

            def sample(_):
                potential = plane_potential_snapshot_V(
                    positions,
                    self._x_axis,
                    self._z_axis,
                    self.superion_charge_C,
                    self.screening_length_m,
                    self.softening_m,
                    self._background_coefficients,
                    source_block_size=self.diagnostics.source_block_size,
                )
                if isinstance(self.background, CylindricalSplineBackground):
                    xx, zz = jnp.meshgrid(self._x_axis, self._z_axis, indexing="xy")
                    targets = jnp.stack(
                        (xx.ravel(), jnp.zeros(xx.size, jnp.float32), zz.ravel()),
                        axis=1,
                    )
                    background_potential, _ = self.background.potential_and_gradient(targets)
                    potential = potential + background_potential.reshape(potential.shape)
                density = plane_cic_density_snapshot_m3(
                    positions,
                    self._x_axis,
                    self._z_axis,
                    self.diagnostics.slice_thickness_m,
                    self.superion_weight,
                )
                return (
                    carry.potential_sum + potential.astype(carry.potential_sum.dtype),
                    carry.density_sum + density.astype(carry.density_sum.dtype),
                    carry.sample_count + 1,
                )

            def no_sample(_):
                return carry.potential_sum, carry.density_sum, carry.sample_count

            potential_sum, density_sum, sample_count = jax.lax.cond(
                should_sample, sample, no_sample, operand=None
            )
        else:
            potential_sum, density_sum, sample_count = (
                carry.potential_sum,
                carry.density_sum,
                carry.sample_count,
            )
        if collision_counts is None:
            collision_counts = (
                jnp.sum(collision_events.candidates, dtype=jnp.int32),
                jnp.sum(collision_events.elastic, dtype=jnp.int32),
                jnp.sum(collision_events.charge_exchange, dtype=jnp.int32),
                jnp.sum(collision_events.null, dtype=jnp.int32),
                jnp.sum(collision_events.out_of_range, dtype=jnp.int32),
                jnp.sum(collision_events.majorant_exceeded, dtype=jnp.int32),
            )
        return _RunCarry(
            next_state,
            pair_forces_N,
            carry.background_forces_N
            if background_forces_N is None
            else background_forces_N,
            carry.fast_forces_N if fast_forces_N is None else fast_forces_N,
            potential_sum,
            density_sum,
            sample_count,
            carry.coincident_count + coincident,
            carry.collision_candidates + collision_counts[0],
            carry.collision_elastic + collision_counts[1],
            carry.collision_charge_exchange + collision_counts[2],
            carry.collision_null + collision_counts[3],
            carry.collision_out_of_range + collision_counts[4],
            carry.collision_majorant_exceeded + collision_counts[5],
            carry.collision_operator_calls
            + jnp.asarray(collision_operator_calls, dtype=jnp.int32),
        )

    def _step(self, carry: _RunCarry, _: None):
        state = carry.state
        key, collision_key, reinjection_key = jax.random.split(
            state.prng_key, 3
        )
        velocity = state.velocities_m_s
        zero_events = jnp.zeros((self.simulation.particle_count,), dtype=bool)
        collision_events = CollisionEvents(
            zero_events, zero_events, zero_events, zero_events, zero_events, zero_events
        )
        if isinstance(self.plasma, FieldDrivenPlasmaSI):
            velocity, legacy_events = self.collision_model.apply(
                collision_key,
                velocity,
                self.plasma.ion_thermal_speed_m_s,
                self.plasma.collision_frequency_s * self.simulation.dt_s,
            )
            collision_events = CollisionEvents(
                legacy_events, zero_events, legacy_events, zero_events, zero_events, zero_events
            )
        elif isinstance(self.plasma, ArgonPresheathPlasmaSI):
            velocity, collision_events = self.collision_model.apply(
                collision_key,
                velocity,
                self.plasma.ion_thermal_speed_m_s,
                self.simulation.dt_s,
            )

        half_velocity = velocity + 0.5 * self.simulation.dt_s * (
            state.forces_N / self.superion_mass_kg
        )
        drifted = state.positions_m + self.simulation.dt_s * half_velocity
        if isinstance(self.plasma, ArgonPresheathPlasmaSI):
            drifted, half_velocity = self._reflect_presheath_side(
                drifted, half_velocity
            )
        radial2 = drifted[:, 0] ** 2 + drifted[:, 1] ** 2
        outside_z = jnp.abs(drifted[:, 2]) > 0.5 * self.simulation.height_m
        outside = (
            outside_z
            if isinstance(self.plasma, ArgonPresheathPlasmaSI)
            else (radial2 > self.simulation.radius_m**2) | outside_z
        )

        dust_count = len(self.dust_particles)
        if dust_count:
            distance2 = jnp.sum(
                (drifted[:, None, :] - self._dust_positions[None, :, :]) ** 2,
                axis=-1,
            )
            inside_each = distance2 <= self._dust_radii[None, :] ** 2
            absorbed = jnp.any(inside_each, axis=1)
            owner = jnp.argmin(distance2, axis=1)
            ownership = jax.nn.one_hot(owner, dust_count, dtype=jnp.float32) * absorbed[:, None]
            absorbed_counts = jnp.sum(ownership, axis=0)
            momentum_velocity = ownership.T @ half_velocity
        else:
            absorbed = jnp.zeros((self.simulation.particle_count,), dtype=bool)
            absorbed_counts = jnp.zeros((0,), jnp.float32)
            momentum_velocity = jnp.zeros((0, 3), jnp.float32)
        invalid = outside | absorbed
        collection_force = (
            self.superion_mass_kg / self.simulation.dt_s * momentum_velocity
        )

        ion_temperature = (
            self.plasma.ion_temperature_K
            if isinstance(self.plasma, MaxwellianPlasmaSI)
            else self.plasma.neutral_temperature_K
        )
        charges = update_dust_charges_C(
            state.dust_charges_C,
            self._dust_modes,
            absorbed_counts,
            self._dust_radii,
            self.superion_charge_C,
            self.plasma.density_m3,
            self.plasma.electron_temperature_K,
            ion_temperature,
            self.plasma.ion_mass_kg,
            self.plasma.flow_speed_m_s,
            self.simulation.dt_s,
        )
        positions, injected_velocity = self._reinject(
            drifted, half_velocity, invalid, reinjection_key
        )
        force, orbit_in, coincident = self._force(positions, charges)
        completed_velocity = injected_velocity + jnp.where(
            invalid[:, None],
            0.0,
            0.5 * self.simulation.dt_s * force / self.superion_mass_kg,
        )
        next_state = SimulationState(
            positions, completed_velocity, force, charges, state.step + 1, key
        )
        orbit, dust_dust, dust_external = self._dust_force_components(
            positions, charges, orbit_in
        )

        next_carry = self._complete_carry(
            carry,
            next_state,
            carry.pair_forces_N,
            coincident,
            collision_events,
            collision_operator_calls=(
                1
                if isinstance(
                    self.plasma, (FieldDrivenPlasmaSI, ArgonPresheathPlasmaSI)
                )
                else 0
            ),
        )
        history = (charges, orbit, collection_force, dust_dust, dust_external)
        return next_carry, history

    def _step_subcycled(self, carry: _RunCarry, _: None):
        state = carry.state
        key, collision_key, subcycle_key = jax.random.split(state.prng_key, 3)
        velocity = state.velocities_m_s
        zero_events = jnp.zeros((self.simulation.particle_count,), dtype=bool)
        collision_events = CollisionEvents(
            zero_events, zero_events, zero_events, zero_events, zero_events, zero_events
        )
        if isinstance(self.plasma, FieldDrivenPlasmaSI):
            velocity, legacy_events = self.collision_model.apply(
                collision_key,
                velocity,
                self.plasma.ion_thermal_speed_m_s,
                self.plasma.collision_frequency_s * self.simulation.dt_s,
            )
            collision_events = CollisionEvents(
                legacy_events, zero_events, legacy_events, zero_events, zero_events, zero_events
            )
        elif isinstance(self.plasma, ArgonPresheathPlasmaSI):
            velocity, collision_events = self.collision_model.apply(
                collision_key,
                velocity,
                self.plasma.ion_thermal_speed_m_s,
                self.simulation.dt_s,
            )

        velocity = velocity + 0.5 * self.simulation.dt_s * (
            (carry.pair_forces_N + carry.background_forces_N)
            / self.superion_mass_kg
        )
        dust_count = len(self.dust_particles)
        ion_temperature = (
            self.plasma.ion_temperature_K
            if isinstance(self.plasma, MaxwellianPlasmaSI)
            else self.plasma.neutral_temperature_K
        )
        initial_momentum = jnp.zeros((dust_count, 3), jnp.float32)

        def fast_substep(inner, substep_index):
            positions, velocities, charges, momentum, fast_force = inner
            half_velocity = velocities + 0.5 * self.charging_dt_s * (
                fast_force / self.superion_mass_kg
            )
            drifted = positions + self.charging_dt_s * half_velocity
            if isinstance(self.plasma, ArgonPresheathPlasmaSI):
                drifted, half_velocity = self._reflect_presheath_side(
                    drifted, half_velocity
                )
            radial2 = drifted[:, 0] ** 2 + drifted[:, 1] ** 2
            outside_z = jnp.abs(drifted[:, 2]) > 0.5 * self.simulation.height_m
            outside = (
                outside_z
                if isinstance(self.plasma, ArgonPresheathPlasmaSI)
                else (radial2 > self.simulation.radius_m**2) | outside_z
            )

            if dust_count:
                distance2 = jnp.sum(
                    (drifted[:, None, :] - self._dust_positions[None, :, :]) ** 2,
                    axis=-1,
                )
                inside_each = distance2 <= self._dust_radii[None, :] ** 2
                absorbed = jnp.any(inside_each, axis=1)
                owner = jnp.argmin(distance2, axis=1)
                ownership = (
                    jax.nn.one_hot(owner, dust_count, dtype=jnp.float32)
                    * absorbed[:, None]
                )
                absorbed_counts = jnp.sum(ownership, axis=0)
                momentum = momentum + ownership.T @ half_velocity
            else:
                absorbed = jnp.zeros((self.simulation.particle_count,), dtype=bool)
                absorbed_counts = jnp.zeros((0,), jnp.float32)
            invalid = outside | absorbed
            charges = update_dust_charges_C(
                charges,
                self._dust_modes,
                absorbed_counts,
                self._dust_radii,
                self.superion_charge_C,
                self.plasma.density_m3,
                self.plasma.electron_temperature_K,
                ion_temperature,
                self.plasma.ion_mass_kg,
                self.plasma.flow_speed_m_s,
                self.charging_dt_s,
            )
            reinjection_key = jax.random.fold_in(subcycle_key, substep_index)
            positions, injected_velocity = self._reinject(
                drifted, half_velocity, invalid, reinjection_key
            )
            next_fast = self._fast_force(positions, charges)
            velocities = injected_velocity + jnp.where(
                invalid[:, None],
                0.0,
                0.5 * self.charging_dt_s * next_fast / self.superion_mass_kg,
            )
            return (
                positions,
                velocities,
                charges,
                momentum,
                next_fast,
            ), None

        initial_inner = (
            state.positions_m,
            velocity,
            state.dust_charges_C,
            initial_momentum,
            carry.fast_forces_N,
        )
        final_inner, _ = jax.lax.scan(
            fast_substep,
            initial_inner,
            jnp.arange(self.charging_substeps, dtype=jnp.int32),
        )
        positions, velocity, charges, momentum, fast_force = final_inner
        pair_force, coincident = self._pair_force(positions)
        background_force = self._background_force(positions)
        completed_velocity = velocity + 0.5 * self.simulation.dt_s * (
            (pair_force + background_force) / self.superion_mass_kg
        )
        force = pair_force + background_force + fast_force
        next_state = SimulationState(
            positions, completed_velocity, force, charges, state.step + 1, key
        )
        orbit_in = self._orbit_force(positions, charges)
        orbit, dust_dust, dust_external = self._dust_force_components(
            positions, charges, orbit_in
        )
        collection_force = self.superion_mass_kg / self.simulation.dt_s * momentum
        next_carry = self._complete_carry(
            carry,
            next_state,
            pair_force,
            coincident,
            collision_events,
            background_force,
            fast_force,
            collision_operator_calls=(
                1
                if isinstance(
                    self.plasma, (FieldDrivenPlasmaSI, ArgonPresheathPlasmaSI)
                )
                else 0
            ),
        )
        history = (charges, orbit, collection_force, dust_dust, dust_external)
        return next_carry, history

    def _step_multirate(self, carry: _RunCarry, _: None):
        """Advance one outer ion-ion step on the merged collision/charge grid."""
        state = carry.state
        next_key, base_step_key = jax.random.split(state.prng_key)
        step_key = jax.random.fold_in(base_step_key, state.step)
        velocity = state.velocities_m_s + 0.5 * self.simulation.dt_s * (
            (carry.pair_forces_N + carry.background_forces_N)
            / self.superion_mass_kg
        )
        dust_count = len(self.dust_particles)
        ion_temperature = (
            self.plasma.ion_temperature_K
            if isinstance(self.plasma, MaxwellianPlasmaSI)
            else self.plasma.neutral_temperature_K
        )
        zero_count = jnp.asarray(0, jnp.int32)

        def fast_segment(inner, schedule):
            (
                positions,
                velocities,
                charges,
                momentum,
                fast_force,
                candidates,
                elastic_count,
                cx_count,
                null_count,
                out_of_range_count,
                majorant_exceeded_count,
                operator_calls,
            ) = inner
            (
                segment_dt,
                collision_index,
                active_collision_index,
                segment_index,
            ) = schedule

            def apply_collision(current_velocity):
                collision_key = jax.random.fold_in(
                    jax.random.fold_in(step_key, 0x434F4C4C), collision_index
                )
                if isinstance(self.plasma, FieldDrivenPlasmaSI):
                    updated, events = self.collision_model.apply(
                        collision_key,
                        current_velocity,
                        self.plasma.ion_thermal_speed_m_s,
                        self.plasma.collision_frequency_s * self.collision_dt_s,
                    )
                    count = jnp.sum(events, dtype=jnp.int32)
                    return updated, (
                        count,
                        zero_count,
                        count,
                        zero_count,
                        zero_count,
                        zero_count,
                        jnp.asarray(1, jnp.int32),
                    )
                if isinstance(self.plasma, ArgonPresheathPlasmaSI):
                    updated, events = self.collision_model.apply(
                        collision_key,
                        current_velocity,
                        self.plasma.ion_thermal_speed_m_s,
                        self.collision_dt_s,
                    )
                    return updated, (
                        jnp.sum(events.candidates, dtype=jnp.int32),
                        jnp.sum(events.elastic, dtype=jnp.int32),
                        jnp.sum(events.charge_exchange, dtype=jnp.int32),
                        jnp.sum(events.null, dtype=jnp.int32),
                        jnp.sum(events.out_of_range, dtype=jnp.int32),
                        jnp.sum(events.majorant_exceeded, dtype=jnp.int32),
                        jnp.asarray(1, jnp.int32),
                    )
                return current_velocity, (
                    zero_count,
                    zero_count,
                    zero_count,
                    zero_count,
                    zero_count,
                    zero_count,
                    zero_count,
                )

            def skip_collision(current_velocity):
                return current_velocity, (
                    zero_count,
                    zero_count,
                    zero_count,
                    zero_count,
                    zero_count,
                    zero_count,
                    zero_count,
                )

            velocities, counts = jax.lax.cond(
                collision_index >= 0,
                apply_collision,
                skip_collision,
                velocities,
            )
            half_velocity = velocities + 0.5 * segment_dt * (
                fast_force / self.superion_mass_kg
            )
            drifted = positions + segment_dt * half_velocity
            if isinstance(self.plasma, ArgonPresheathPlasmaSI):
                drifted, half_velocity = self._reflect_presheath_side(
                    drifted, half_velocity
                )
            radial2 = drifted[:, 0] ** 2 + drifted[:, 1] ** 2
            outside_z = jnp.abs(drifted[:, 2]) > 0.5 * self.simulation.height_m
            outside = (
                outside_z
                if isinstance(self.plasma, ArgonPresheathPlasmaSI)
                else (radial2 > self.simulation.radius_m**2) | outside_z
            )

            if dust_count:
                distance2 = jnp.sum(
                    (drifted[:, None, :] - self._dust_positions[None, :, :]) ** 2,
                    axis=-1,
                )
                inside_each = distance2 <= self._dust_radii[None, :] ** 2
                absorbed = jnp.any(inside_each, axis=1)
                owner = jnp.argmin(distance2, axis=1)
                ownership = (
                    jax.nn.one_hot(owner, dust_count, dtype=jnp.float32)
                    * absorbed[:, None]
                )
                absorbed_counts = jnp.sum(ownership, axis=0)
                momentum = momentum + ownership.T @ half_velocity
            else:
                absorbed = jnp.zeros((self.simulation.particle_count,), dtype=bool)
                absorbed_counts = jnp.zeros((0,), jnp.float32)
            invalid = outside | absorbed
            charges = update_dust_charges_C(
                charges,
                self._dust_modes,
                absorbed_counts,
                self._dust_radii,
                self.superion_charge_C,
                self.plasma.density_m3,
                self.plasma.electron_temperature_K,
                ion_temperature,
                self.plasma.ion_mass_kg,
                self.plasma.flow_speed_m_s,
                segment_dt,
            )
            reinjection_key = jax.random.fold_in(
                jax.random.fold_in(
                    jax.random.fold_in(step_key, 0x5245494E),
                    active_collision_index,
                ),
                segment_index,
            )
            positions, injected_velocity = self._reinject(
                drifted, half_velocity, invalid, reinjection_key
            )
            next_fast = self._fast_force(positions, charges)
            velocities = injected_velocity + jnp.where(
                invalid[:, None],
                0.0,
                0.5 * segment_dt * next_fast / self.superion_mass_kg,
            )
            return (
                positions,
                velocities,
                charges,
                momentum,
                next_fast,
                candidates + counts[0],
                elastic_count + counts[1],
                cx_count + counts[2],
                null_count + counts[3],
                out_of_range_count + counts[4],
                majorant_exceeded_count + counts[5],
                operator_calls + counts[6],
            ), None

        initial_inner = (
            state.positions_m,
            velocity,
            state.dust_charges_C,
            jnp.zeros((dust_count, 3), jnp.float32),
            carry.fast_forces_N,
            zero_count,
            zero_count,
            zero_count,
            zero_count,
            zero_count,
            zero_count,
            zero_count,
        )
        final_inner, _ = jax.lax.scan(
            fast_segment,
            initial_inner,
            (
                self._segment_durations_s,
                self._segment_collision_indices,
                self._segment_active_collision_indices,
                self._segment_indices,
            ),
        )
        (
            positions,
            velocity,
            charges,
            momentum,
            fast_force,
            candidate_count,
            elastic_count,
            cx_count,
            null_count,
            out_of_range_count,
            majorant_exceeded_count,
            operator_calls,
        ) = final_inner
        pair_force, coincident = self._pair_force(positions)
        background_force = self._background_force(positions)
        completed_velocity = velocity + 0.5 * self.simulation.dt_s * (
            (pair_force + background_force) / self.superion_mass_kg
        )
        force = pair_force + background_force + fast_force
        next_state = SimulationState(
            positions, completed_velocity, force, charges, state.step + 1, next_key
        )
        orbit_in = self._orbit_force(positions, charges)
        orbit, dust_dust, dust_external = self._dust_force_components(
            positions, charges, orbit_in
        )
        collection_force = self.superion_mass_kg / self.simulation.dt_s * momentum
        zero_events = jnp.zeros((self.simulation.particle_count,), dtype=bool)
        empty_events = CollisionEvents(
            zero_events, zero_events, zero_events, zero_events, zero_events, zero_events
        )
        next_carry = self._complete_carry(
            carry,
            next_state,
            pair_force,
            coincident,
            empty_events,
            background_force,
            fast_force,
            (
                candidate_count,
                elastic_count,
                cx_count,
                null_count,
                out_of_range_count,
                majorant_exceeded_count,
            ),
            operator_calls,
        )
        history = (charges, orbit, collection_force, dust_dust, dust_external)
        return next_carry, history

    def run(self, initial_state: SimulationState | None = None) -> SimulationResult:
        state = initial_state if initial_state is not None else self.initial_state()
        expected_shape = (self.simulation.particle_count, 3)
        if state.positions_m.shape != expected_shape or state.velocities_m_s.shape != expected_shape:
            raise ValueError(f"initial_state particle arrays must have shape {expected_shape}")
        if state.dust_charges_C.shape != (len(self.dust_particles),):
            raise ValueError("initial_state dust charge count does not match configuration")

        accumulator_dtype = jnp.float64 if jax.config.x64_enabled else jnp.float32
        grid_shape = (
            (self.diagnostics.nz, self.diagnostics.nx)
            if self.diagnostics is not None else (0, 0)
        )
        initial_pair_force, initial_coincident = self._pair_force(state.positions_m)
        initial_background_force = self._background_force(state.positions_m)
        initial_fast_force = self._fast_force(
            state.positions_m, state.dust_charges_C
        )
        initial_orbit_in = self._orbit_force(
            state.positions_m, state.dust_charges_C
        )
        carry = _RunCarry(
            state,
            initial_pair_force,
            initial_background_force,
            initial_fast_force,
            jnp.zeros(grid_shape, accumulator_dtype),
            jnp.zeros(grid_shape, accumulator_dtype),
            jnp.asarray(0, jnp.int32),
            jnp.asarray(0, jnp.int32),
            jnp.asarray(0, jnp.int32),
            jnp.asarray(0, jnp.int32),
            jnp.asarray(0, jnp.int32),
            jnp.asarray(0, jnp.int32),
            jnp.asarray(0, jnp.int32),
            jnp.asarray(0, jnp.int32),
            jnp.asarray(0, jnp.int32),
        )
        initial_orbit, initial_dust_dust, initial_external = self._dust_force_components(
            state.positions_m, state.dust_charges_C, initial_orbit_in
        )
        zero_collection = jnp.zeros_like(initial_orbit)

        base_chunk = self.simulation.step_chunk_size
        for interval in (
            self.output.checkpoint_interval,
            self.output.trajectory_interval,
        ):
            if interval is not None:
                base_chunk = math.gcd(base_chunk, interval)
        histories: list[tuple[np.ndarray, ...]] = []
        trajectory_steps: list[int] = []
        trajectory_positions: list[np.ndarray] = []
        collision_totals = [0, 0, 0, 0, 0, 0, 0]
        completed = 0
        while completed < self.simulation.steps:
            chunk = min(base_chunk, self.simulation.steps - completed)
            if chunk not in self._compiled_chunks:
                step_function = (
                    self._step_multirate
                    if self.simulation.collision_subcycling is not None
                    else (
                        self._step_subcycled
                        if self.simulation.charging_subcycling is not None
                        else self._step
                    )
                )
                def run_chunk(value, *, _length=chunk):
                    return jax.lax.scan(step_function, value, xs=None, length=_length)
                self._compiled_chunks[chunk] = jax.jit(run_chunk)
            carry, chunk_history = self._compiled_chunks[chunk](carry)
            chunk_collision_counts = [
                int(np.asarray(value))
                for value in jax.device_get(
                    (
                        carry.collision_candidates,
                        carry.collision_elastic,
                        carry.collision_charge_exchange,
                        carry.collision_null,
                        carry.collision_out_of_range,
                        carry.collision_majorant_exceeded,
                        carry.collision_operator_calls,
                    )
                )
            ]
            if chunk_collision_counts[4] > 0:
                raise RuntimeError(
                    "Ar+/Ar collision energy exceeded the 10 keV table range"
                )
            collision_totals = [
                total + value for total, value in zip(collision_totals, chunk_collision_counts)
            ]
            zero_count = jnp.asarray(0, jnp.int32)
            carry = carry._replace(
                collision_candidates=zero_count,
                collision_elastic=zero_count,
                collision_charge_exchange=zero_count,
                collision_null=zero_count,
                collision_out_of_range=zero_count,
                collision_majorant_exceeded=zero_count,
                collision_operator_calls=zero_count,
            )
            host_history = tuple(np.asarray(x) for x in jax.device_get(chunk_history))
            histories.append(host_history)
            completed += chunk
            global_step = int(np.asarray(jax.device_get(carry.state.step)))
            if self.output.trajectory_interval is not None and global_step % self.output.trajectory_interval == 0:
                trajectory_steps.append(global_step)
                trajectory_positions.append(
                    np.asarray(jax.device_get(carry.state.positions_m), dtype=np.float32)
                )
            if self.output.checkpoint_interval is not None and global_step % self.output.checkpoint_interval == 0:
                save_checkpoint(
                    self.output.checkpoint_path,
                    carry.state,
                    self._checkpoint_background_array(),
                    self._metadata(),
                )

        if histories:
            concatenated = tuple(
                np.concatenate([chunk[index] for chunk in histories], axis=0)
                for index in range(5)
            )
        else:
            dust_shape = (0, len(self.dust_particles))
            concatenated = (
                np.empty(dust_shape, np.float32),
                np.empty(dust_shape + (3,), np.float32),
                np.empty(dust_shape + (3,), np.float32),
                np.empty(dust_shape + (3,), np.float32),
                np.empty(dust_shape + (3,), np.float32),
            )
        start_step = int(np.asarray(jax.device_get(state.step)))
        times = (start_step + np.arange(self.simulation.steps + 1)) * self.simulation.dt_s
        charges = np.concatenate(
            (np.asarray(jax.device_get(state.dust_charges_C))[None, :], concatenated[0]), axis=0
        )
        orbit = np.concatenate((np.asarray(jax.device_get(initial_orbit))[None, ...], concatenated[1]), axis=0)
        collection = np.concatenate((np.asarray(zero_collection)[None, ...], concatenated[2]), axis=0)
        dust_dust = np.concatenate((np.asarray(initial_dust_dust)[None, ...], concatenated[3]), axis=0)
        external = np.concatenate((np.asarray(initial_external)[None, ...], concatenated[4]), axis=0)

        diagnostic_files: tuple[Path, ...] = ()
        if self.diagnostics is not None:
            host_carry = jax.device_get(carry)
            sample_count = int(np.asarray(host_carry.sample_count))
            if sample_count == 0:
                raise RuntimeError("diagnostic configuration produced no samples")
            first = max(
                self.diagnostics.start_step,
                start_step + 1,
            )
            remainder = (first - self.diagnostics.start_step) % self.diagnostics.interval
            if remainder:
                first += self.diagnostics.interval - remainder
            sample_steps = np.arange(
                first,
                start_step + self.simulation.steps + 1,
                self.diagnostics.interval,
                dtype=np.int64,
            )
            diagnostic_files = write_plane_diagnostics(
                self.diagnostics,
                np.asarray(jax.device_get(self._x_axis)),
                np.asarray(jax.device_get(self._z_axis)),
                np.asarray(host_carry.potential_sum) / sample_count,
                np.asarray(host_carry.density_sum) / sample_count,
                sample_steps,
                self.simulation.dt_s,
                self._metadata(),
            )
        if self.output.checkpoint_path is not None and self.output.checkpoint_interval is None:
            save_checkpoint(
                self.output.checkpoint_path,
                carry.state,
                self._checkpoint_background_array(),
                self._metadata(),
            )
        return SimulationResult(
            final_state=carry.state,
            times_s=times,
            dust_charges_C=charges,
            dust_orbit_forces_N=orbit,
            dust_collection_forces_N=collection,
            dust_dust_forces_N=dust_dust,
            dust_external_forces_N=external,
            coincident_pair_count=int(np.asarray(jax.device_get(carry.coincident_count + initial_coincident))),
            diagnostic_files=diagnostic_files,
            trajectory_steps=np.asarray(trajectory_steps, dtype=np.int64),
            trajectory_positions_m=(
                np.stack(trajectory_positions) if trajectory_positions else None
            ),
            collision_statistics=CollisionStatistics(
                candidate_count=collision_totals[0],
                elastic_count=collision_totals[1],
                charge_exchange_count=collision_totals[2],
                null_count=collision_totals[3],
                out_of_range_count=collision_totals[4],
                majorant_exceeded_count=collision_totals[5],
                operator_call_count=collision_totals[6],
            ),
        )

    def run_from_checkpoint(self, path: Path | str) -> SimulationResult:
        """Validate the saved background and continue from a checkpoint."""
        state, coefficients, metadata = load_checkpoint(path)
        if isinstance(self.collision_model, ArgonNullCollisionModel):
            expected_majorant = {
                "maximum_relative_speed_m_s": (
                    self.collision_model.maximum_relative_speed_m_s
                ),
                "maximum_energy_eV": (
                    self.collision_model.maximum_majorant_energy_eV
                ),
                "maximum_frequency_s": self.collision_model.maximum_frequency_s,
            }
            if metadata.get("collision_majorant") != expected_majorant:
                raise ValueError("checkpoint collision majorant does not match")
        saved_collision_subcycling = metadata.get("collision_subcycling")
        collision_enabled = self.simulation.collision_subcycling is not None
        if saved_collision_subcycling is None:
            if collision_enabled:
                raise ValueError(
                    "legacy checkpoint can only resume with collision subcycling disabled"
                )
        else:
            saved_durations = np.asarray(
                saved_collision_subcycling.get(
                    "segment_durations_s", [self.simulation.dt_s]
                ),
                dtype=np.float64,
            )
            durations_match = (
                saved_durations.shape == self._segment_durations_host.shape
                and np.allclose(
                    saved_durations,
                    self._segment_durations_host,
                    rtol=1.0e-12,
                    atol=0.0,
                )
            )
            if (
                bool(saved_collision_subcycling.get("enabled"))
                != collision_enabled
                or metadata.get("simulation", {}).get("collision_subcycling")
                != asdict(self.simulation).get("collision_subcycling")
                or int(saved_collision_subcycling.get("resolved_substeps", 1))
                != self.collision_substeps
                or not math.isclose(
                    float(
                        saved_collision_subcycling.get(
                            "dt_collision_s", self.simulation.dt_s
                        )
                    ),
                    self.collision_dt_s,
                )
                or int(saved_collision_subcycling.get("integration_segments", 1))
                != self.integration_segments
                or not durations_match
            ):
                raise ValueError(
                    "checkpoint collision subcycling configuration does not match"
                )
        saved_subcycling = metadata.get("charging_subcycling")
        enabled = self.simulation.charging_subcycling is not None
        if saved_subcycling is None:
            if enabled:
                raise ValueError(
                    "legacy checkpoint can only resume with charging subcycling disabled"
                )
        elif (
            bool(saved_subcycling.get("enabled")) != enabled
            or metadata.get("simulation", {}).get("charging_subcycling")
            != asdict(self.simulation).get("charging_subcycling")
            or int(saved_subcycling.get("resolved_substeps", 1))
            != self.charging_substeps
            or not math.isclose(
                float(saved_subcycling.get("dt_charge_s", self.simulation.dt_s)),
                self.charging_dt_s,
            )
        ):
            raise ValueError("checkpoint charging subcycling configuration does not match")
        if isinstance(self.background, CylindricalSplineBackground):
            saved = metadata.get("background", {}).get("fingerprint")
            if saved != self.background.fingerprint:
                raise ValueError("checkpoint presheath background does not match")
        elif coefficients.shape != (15,) or not np.allclose(
            coefficients, self.background.coefficients, rtol=1.0e-12, atol=0.0
        ):
            raise ValueError("checkpoint background coefficients do not match this configuration")
        return self.run(state)

    def _metadata(self) -> dict:
        return {
            "plasma": asdict(self.plasma),
            "simulation": asdict(self.simulation),
            "dust_particles": [asdict(dust) for dust in self.dust_particles],
            "superion_weight": self.superion_weight,
            "superion_charge_C": self.superion_charge_C,
            "superion_mass_kg": self.superion_mass_kg,
            "screening_length_m": self.screening_length_m,
            "softening_m": self.softening_m,
            "time_splitting_version": 1,
            "collision_majorant": (
                {
                    "maximum_relative_speed_m_s": (
                        self.collision_model.maximum_relative_speed_m_s
                    ),
                    "maximum_energy_eV": (
                        self.collision_model.maximum_majorant_energy_eV
                    ),
                    "maximum_frequency_s": self.collision_model.maximum_frequency_s,
                }
                if isinstance(self.collision_model, ArgonNullCollisionModel)
                else None
            ),
            "collision_subcycling": {
                "enabled": self.simulation.collision_subcycling is not None,
                "resolved_substeps": self.collision_substeps,
                "dt_collision_s": self.collision_dt_s,
                "integration_segments": self.integration_segments,
                "maximum_fast_dt_s": self.maximum_fast_dt_s,
                "segment_durations_s": self._segment_durations_host.tolist(),
            },
            "charging_subcycling": {
                "enabled": self.simulation.charging_subcycling is not None,
                "resolved_substeps": self.charging_substeps,
                "dt_charge_s": self.charging_dt_s,
            },
            "background": (
                {
                    "kind": "cylindrical_spline",
                    "fingerprint": self.background.fingerprint,
                }
                if isinstance(self.background, CylindricalSplineBackground)
                else {
                    "kind": "polynomial",
                    "coefficients": self.background.coefficients.tolist(),
                    "relative_rms_error": self.background.relative_rms_error,
                    "sample_count": self.background.sample_count,
                    "cache_key": self.background.cache_key,
                }
            ),
        }


__all__ = ["OpenDustJAX"]
