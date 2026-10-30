"""Single-NVIDIA-GPU OpenDust simulation engine using JAX and CUDA FFI."""

from __future__ import annotations

from dataclasses import asdict
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
from opendust_jax.collisions import CollisionModel, LegacyChargeExchangeCollision
from opendust_jax.cuda_pair_forces import (
    cuda_pair_forces_available,
    external_forces_cuda,
    softened_yukawa_forces_cuda_with_stats,
)
from opendust_jax.diagnostics import (
    diagnostic_axes,
    plane_cic_density_snapshot_m3,
    plane_potential_snapshot_V,
    write_plane_diagnostics,
)
from opendust_jax.external_forces import external_forces_jax
from opendust_jax.models import (
    BOLTZMANN_J_K,
    COULOMB_CONSTANT,
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
from opendust_jax.pair_forces import softened_yukawa_forces
from opendust_jax.physics import (
    background_force_N,
    dust_dust_forces_N,
    update_dust_charges_C,
)


class _RunCarry(NamedTuple):
    state: SimulationState
    potential_sum: jax.Array
    density_sum: jax.Array
    sample_count: jax.Array
    coincident_count: jax.Array


def _normal_pdf(values: np.ndarray, mean: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((values - mean) / sigma) ** 2) / (
        math.sqrt(2.0 * math.pi) * sigma
    )


def _axial_pdf(
    values: np.ndarray, plasma: PlasmaSI, thermal_speed: float, flow_speed: float
) -> np.ndarray:
    if isinstance(plasma, MaxwellianPlasmaSI) or abs(flow_speed) < 1.0e-12 * thermal_speed:
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
        background: PolynomialBackgroundConfig | None = None,
        diagnostics: PlaneDiagnosticConfig | None = None,
        output: OutputConfig | None = None,
        collision_model: CollisionModel | None = None,
    ) -> None:
        self.plasma = plasma
        self.simulation = simulation
        self.dust_particles = tuple(dust_particles)
        self.background_config = background or PolynomialBackgroundConfig()
        self.diagnostics = diagnostics
        self.output = output or OutputConfig()
        self.collision_model = collision_model or LegacyChargeExchangeCollision()
        validate_system(plasma, simulation, self.dust_particles, diagnostics)
        if simulation.force_backend == "cuda" and not cuda_pair_forces_available():
            raise RuntimeError(
                "The OpenDustJAX CUDA FFI library is unavailable. Build it with "
                "performance/JAX/build_cuda_pair_forces.py on an NVIDIA GPU host."
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
        self._background_coefficients = jnp.asarray(
            self.background.coefficients, dtype=jnp.float32
        )
        if diagnostics is not None:
            self._x_axis, self._z_axis = diagnostic_axes(
                simulation.radius_m, simulation.height_m, diagnostics
            )
        else:
            self._x_axis = self._z_axis = jnp.zeros((0,), jnp.float32)

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

    def _force(
        self, positions: jax.Array, dust_charges: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        prefactor = COULOMB_CONSTANT * self.superion_charge_C**2
        electric_field = (
            self.plasma.electric_field_V_m
            if isinstance(self.plasma, FieldDrivenPlasmaSI)
            else 0.0
        )
        if self.simulation.force_backend == "cuda":
            pair, coincident = softened_yukawa_forces_cuda_with_stats(
                positions, prefactor, self.screening_length_m, self.softening_m
            )
            external, orbit = external_forces_cuda(
                positions,
                self._dust_positions,
                dust_charges,
                self._background_coefficients,
                self.superion_charge_C,
                self.screening_length_m,
                electric_field,
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
            external, orbit = external_forces_jax(
                positions,
                self._dust_positions,
                dust_charges,
                self._background_coefficients,
                self.superion_charge_C,
                self.screening_length_m,
                electric_field,
            )
        return pair + external, orbit, coincident

    def initial_state(self) -> SimulationState:
        positions, velocities = self._initial_arrays()
        charges = jnp.asarray(
            [d.initial_charge_C for d in self.dust_particles], dtype=jnp.float32
        )
        force, _, _ = self._force(jnp.asarray(positions), charges)
        return SimulationState(
            jnp.asarray(positions),
            jnp.asarray(velocities),
            force,
            charges,
            jnp.asarray(0, jnp.int32),
            jax.random.PRNGKey(self.simulation.seed),
        )

    def _reinject(
        self, positions: jax.Array, velocities: jax.Array, mask: jax.Array, key: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
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

    def _dust_force_components(
        self, positions: jax.Array, charges: jax.Array, orbit_in: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        orbit = orbit_in + background_force_N(
            self._dust_positions,
            charges,
            self._background_coefficients,
            self.screening_length_m,
            self.superion_charge_C,
        )
        dust_dust = dust_dust_forces_N(self._dust_positions, charges)
        electric_field = (
            self.plasma.electric_field_V_m
            if isinstance(self.plasma, FieldDrivenPlasmaSI)
            else 0.0
        )
        external = jnp.zeros_like(self._dust_positions).at[:, 2].set(
            charges * electric_field
        )
        return orbit, dust_dust, external

    def _step(self, carry: _RunCarry, _: None):
        state = carry.state
        key, collision_key, reinjection_key = jax.random.split(
            state.prng_key, 3
        )
        velocity = state.velocities_m_s
        if isinstance(self.plasma, FieldDrivenPlasmaSI):
            velocity, _ = self.collision_model.apply(
                collision_key,
                velocity,
                self.plasma.ion_thermal_speed_m_s,
                self.plasma.collision_frequency_s * self.simulation.dt_s,
            )

        half_velocity = velocity + 0.5 * self.simulation.dt_s * (
            state.forces_N / self.superion_mass_kg
        )
        drifted = state.positions_m + self.simulation.dt_s * half_velocity
        radial2 = drifted[:, 0] ** 2 + drifted[:, 1] ** 2
        outside = (radial2 > self.simulation.radius_m**2) | (
            jnp.abs(drifted[:, 2]) > 0.5 * self.simulation.height_m
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
        next_carry = _RunCarry(
            next_state,
            potential_sum,
            density_sum,
            sample_count,
            carry.coincident_count + coincident,
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
        carry = _RunCarry(
            state,
            jnp.zeros(grid_shape, accumulator_dtype),
            jnp.zeros(grid_shape, accumulator_dtype),
            jnp.asarray(0, jnp.int32),
            jnp.asarray(0, jnp.int32),
        )
        _, initial_orbit_in, initial_coincident = self._force(
            state.positions_m, state.dust_charges_C
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
        completed = 0
        compiled_chunks: dict[int, object] = {}
        while completed < self.simulation.steps:
            chunk = min(base_chunk, self.simulation.steps - completed)
            if chunk not in compiled_chunks:
                def run_chunk(value, *, _length=chunk):
                    return jax.lax.scan(self._step, value, xs=None, length=_length)
                compiled_chunks[chunk] = jax.jit(run_chunk)
            carry, chunk_history = compiled_chunks[chunk](carry)
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
                    self.background.coefficients,
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
                self.background.coefficients,
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
        )

    def run_from_checkpoint(self, path: Path | str) -> SimulationResult:
        """Validate the saved background and continue from a checkpoint."""
        state, coefficients, _ = load_checkpoint(path)
        if coefficients.shape != (15,) or not np.allclose(
            coefficients, self.background.coefficients, rtol=1.0e-12, atol=0.0
        ):
            raise ValueError(
                "checkpoint background coefficients do not match this configuration"
            )
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
            "background": {
                "coefficients": self.background.coefficients.tolist(),
                "relative_rms_error": self.background.relative_rms_error,
                "sample_count": self.background.sample_count,
                "cache_key": self.background.cache_key,
            },
        }


__all__ = ["OpenDustJAX"]
