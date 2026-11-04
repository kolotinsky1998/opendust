"""Preparation and interpolation of non-uniform presheath backgrounds."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from opendust_jax.collisions import ArgonNullCollisionModel
from opendust_jax.models import (
    COULOMB_CONSTANT,
    ELEMENTARY_CHARGE_C,
    ArgonPresheathPlasmaSI,
    PresheathEquilibrationConfig,
    SimulationConfigSI,
)


@dataclass(frozen=True)
class CylindricalSplineBackground:
    """Axisymmetric compensating potential tabulated on a uniform r-z grid."""

    r_m: np.ndarray
    z_m: np.ndarray
    density_m3: np.ndarray
    ion_potential_V: np.ndarray
    potential_V: np.ndarray
    fingerprint: str
    convergence_history: np.ndarray
    metadata: dict
    axial_flux_m2_s: np.ndarray | None = None

    @property
    def axial_velocity_m_s(self) -> np.ndarray:
        """Cell-averaged axial velocity; old artifacts produce a zero field."""
        if self.axial_flux_m2_s is None:
            return np.zeros_like(self.density_m3, dtype=np.float64)
        return np.divide(
            self.axial_flux_m2_s,
            self.density_m3,
            out=np.zeros_like(self.density_m3, dtype=np.float64),
            where=self.density_m3 > 0.0,
        )

    def potential_and_gradient(
        self, positions_m: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        return cylindrical_cubic_value_and_gradient(
            positions_m,
            jnp.asarray(self.r_m, jnp.float32),
            jnp.asarray(self.z_m, jnp.float32),
            jnp.asarray(self.potential_V, jnp.float32),
        )

    def force_N(self, positions_m: jax.Array, charges_C: float | jax.Array) -> jax.Array:
        _, gradient = self.potential_and_gradient(positions_m)
        return -jnp.asarray(charges_C)[..., None] * gradient


def _cubic_weights(t: jax.Array) -> tuple[jax.Array, jax.Array]:
    t2, t3 = t * t, t * t * t
    weights = jnp.stack(
        (
            -0.5 * t + t2 - 0.5 * t3,
            1.0 - 2.5 * t2 + 1.5 * t3,
            0.5 * t + 2.0 * t2 - 1.5 * t3,
            -0.5 * t2 + 0.5 * t3,
        ),
        axis=-1,
    )
    derivatives = jnp.stack(
        (
            -0.5 + 2.0 * t - 1.5 * t2,
            -5.0 * t + 4.5 * t2,
            0.5 + 4.0 * t - 4.5 * t2,
            -t + 1.5 * t2,
        ),
        axis=-1,
    )
    return weights, derivatives


def cylindrical_cubic_value_and_gradient(
    positions_m: jax.Array,
    r_axis_m: jax.Array,
    z_axis_m: jax.Array,
    values_V: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Evaluate a uniform tensor-product cubic interpolant and its gradient."""
    positions = jnp.asarray(positions_m, dtype=jnp.float32)
    x, y, z = positions[..., 0], positions[..., 1], positions[..., 2]
    radius = jnp.sqrt(x * x + y * y)
    dr, dz = r_axis_m[1] - r_axis_m[0], z_axis_m[1] - z_axis_m[0]
    ur = jnp.clip((radius - r_axis_m[0]) / dr, 0.0, r_axis_m.size - 1.000001)
    uz = jnp.clip((z - z_axis_m[0]) / dz, 0.0, z_axis_m.size - 1.000001)
    ir, iz = jnp.floor(ur).astype(jnp.int32), jnp.floor(uz).astype(jnp.int32)
    wr, dwr = _cubic_weights(ur - ir)
    wz, dwz = _cubic_weights(uz - iz)
    r_indices = jnp.clip(ir[..., None] + jnp.arange(-1, 3), 0, r_axis_m.size - 1)
    z_indices = jnp.clip(iz[..., None] + jnp.arange(-1, 3), 0, z_axis_m.size - 1)
    patch = values_V[z_indices[..., :, None], r_indices[..., None, :]]
    value = jnp.einsum("...i,...ij,...j->...", wz, patch, wr)
    d_radius = jnp.einsum("...i,...ij,...j->...", wz, patch, dwr) / dr
    d_z = jnp.einsum("...i,...ij,...j->...", dwz, patch, wr) / dz
    inverse_radius = jnp.where(radius > 0.0, 1.0 / radius, 0.0)
    gradient = jnp.stack(
        (d_radius * x * inverse_radius, d_radius * y * inverse_radius, d_z), axis=-1
    )
    return value, gradient


def _fingerprint(
    plasma: ArgonPresheathPlasmaSI,
    simulation: SimulationConfigSI,
    equilibration: PresheathEquilibrationConfig,
    softening_m: float,
    checksums: tuple[str, str],
) -> str:
    payload = {
        "format_version": 1,
        "plasma": asdict(plasma),
        "geometry": {
            "radius_m": simulation.radius_m,
            "height_m": simulation.height_m,
            "particle_count": simulation.particle_count,
            "dt_s": simulation.dt_s,
            "seed": simulation.seed,
        },
        "equilibration": {
            key: value
            for key, value in asdict(equilibration).items()
            if key != "output_prefix"
        },
        "softening_m": softening_m,
        "boundary_model": "upstream_reinjection_specular_side",
        "cross_section_sha256": checksums,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:24]


def save_presheath_background(
    background: CylindricalSplineBackground, prefix: Path | str
) -> tuple[Path, Path]:
    base = Path(prefix).expanduser()
    base.parent.mkdir(parents=True, exist_ok=True)
    npz_path = base.with_name(base.name + "_background.npz")
    json_path = base.with_name(base.name + "_background.json")
    temporary_npz = npz_path.with_name(npz_path.name + ".tmp.npz")
    temporary_json = json_path.with_name(json_path.name + ".tmp")
    axial_flux = (
        np.zeros_like(background.density_m3)
        if background.axial_flux_m2_s is None
        else background.axial_flux_m2_s
    )
    np.savez_compressed(
        temporary_npz,
        r_m=background.r_m,
        z_m=background.z_m,
        density_m3=background.density_m3,
        ion_potential_V=background.ion_potential_V,
        potential_V=background.potential_V,
        axial_flux_m2_s=axial_flux,
        convergence_history=background.convergence_history,
        fingerprint=np.asarray(background.fingerprint),
        metadata_json=np.asarray(json.dumps(background.metadata, sort_keys=True)),
    )
    temporary_npz.replace(npz_path)
    temporary_json.write_text(
        json.dumps(
            {
                "format_version": 2,
                "fingerprint": background.fingerprint,
                "units": {
                    "r_m": "m", "z_m": "m", "density_m3": "m^-3",
                    "ion_potential_V": "V", "potential_V": "V",
                    "axial_flux_m2_s": "m^-2 s^-1",
                    "axial_velocity_m_s": "m s^-1",
                },
                **background.metadata,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    temporary_json.replace(json_path)
    return npz_path, json_path


def load_presheath_background(path: Path | str) -> CylindricalSplineBackground:
    source = Path(path).expanduser()
    if not source.is_file():
        candidate = source.with_name(source.name + "_background.npz")
        if candidate.is_file():
            source = candidate
    with np.load(source, allow_pickle=False) as data:
        density = np.asarray(data["density_m3"], dtype=np.float64)
        axial_flux = (
            np.asarray(data["axial_flux_m2_s"], dtype=np.float64)
            if "axial_flux_m2_s" in data.files
            else np.zeros_like(density)
        )
        return CylindricalSplineBackground(
            r_m=np.asarray(data["r_m"], dtype=np.float64),
            z_m=np.asarray(data["z_m"], dtype=np.float64),
            density_m3=density,
            ion_potential_V=np.asarray(data["ion_potential_V"], dtype=np.float64),
            potential_V=np.asarray(data["potential_V"], dtype=np.float64),
            fingerprint=str(data["fingerprint"]),
            convergence_history=np.asarray(data["convergence_history"], dtype=np.float64),
            metadata=json.loads(str(data["metadata_json"])),
            axial_flux_m2_s=axial_flux,
        )


def _field_values(plasma: ArgonPresheathPlasmaSI, z: jax.Array) -> jax.Array:
    field = plasma.electric_field
    lower = -jnp.inf if field.z_min_m is None else field.z_min_m
    upper = jnp.inf if field.z_max_m is None else field.z_max_m
    bounded = jnp.clip(z, lower, upper)
    return field.E0_V_m + field.alpha_V_m2 * bounded + field.beta_V_m3 * bounded**2


def _flux_table(plasma: ArgonPresheathPlasmaSI) -> tuple[jax.Array, jax.Array]:
    thermal, flow = plasma.ion_thermal_speed_m_s, plasma.flow_speed_m_s
    values = np.linspace(0.0, flow + 10.0 * thermal, 8193)
    weights = values * np.exp(-0.5 * ((values - flow) / thermal) ** 2)
    increments = 0.5 * (weights[1:] + weights[:-1]) * np.diff(values)
    cdf = np.concatenate(([0.0], np.cumsum(increments)))
    cdf /= cdf[-1]
    cdf[-1] = 1.0
    return jnp.asarray(values, jnp.float32), jnp.asarray(cdf, jnp.float32)


def _relative_change(current: np.ndarray, previous: np.ndarray) -> float:
    scale = max(float(np.linalg.norm(previous)), np.finfo(float).tiny)
    return float(np.linalg.norm(current - previous) / scale)


def _direct_potential(
    positions_m: np.ndarray,
    r_m: np.ndarray,
    z_m: np.ndarray,
    charge_C: float,
    screening_length_m: float,
    softening_m: float,
    block_size: int,
) -> np.ndarray:
    rr, zz = np.meshgrid(r_m, z_m, indexing="xy")
    targets = np.stack((rr.ravel(), np.zeros(rr.size), zz.ravel()), axis=1)
    result = np.zeros(targets.shape[0], dtype=np.float64)
    for start in range(0, len(positions_m), block_size):
        block = positions_m[start : start + block_size]
        distance = np.linalg.norm(targets[:, None, :] - block[None, :, :], axis=-1)
        result += np.sum(np.exp(-distance / screening_length_m) / (distance + softening_m), axis=1)
    return (COULOMB_CONSTANT * charge_C * result).reshape((len(z_m), len(r_m)))


def _axisymmetric_node_density(
    positions_m: np.ndarray,
    r_m: np.ndarray,
    z_m: np.ndarray,
    superion_weight: float,
    radius_m: float,
    height_m: float,
    particle_values: np.ndarray | None = None,
) -> np.ndarray:
    radial = np.linalg.norm(positions_m[:, :2], axis=1)
    dr, dz = r_m[1] - r_m[0], z_m[1] - z_m[0]
    gr = np.clip(radial / dr, 0.0, len(r_m) - 1.0)
    gz = np.clip((positions_m[:, 2] - z_m[0]) / dz, 0.0, len(z_m) - 1.0)
    ir = np.minimum(np.floor(gr).astype(int), len(r_m) - 2)
    iz = np.minimum(np.floor(gz).astype(int), len(z_m) - 2)
    tr, tz = gr - ir, gz - iz
    weights = np.zeros((len(z_m), len(r_m)), dtype=np.float64)
    values = np.ones(len(positions_m)) if particle_values is None else particle_values
    np.add.at(weights, (iz, ir), values * (1.0 - tr) * (1.0 - tz))
    np.add.at(weights, (iz, ir + 1), values * tr * (1.0 - tz))
    np.add.at(weights, (iz + 1, ir), values * (1.0 - tr) * tz)
    np.add.at(weights, (iz + 1, ir + 1), values * tr * tz)
    r_edges = np.concatenate(([0.0], 0.5 * (r_m[1:] + r_m[:-1]), [radius_m]))
    z_edges = np.concatenate(([-0.5 * height_m], 0.5 * (z_m[1:] + z_m[:-1]), [0.5 * height_m]))
    volumes = (
        math.pi * (r_edges[1:] ** 2 - r_edges[:-1] ** 2)[None, :]
        * np.diff(z_edges)[:, None]
    )
    return weights * superion_weight / volumes


def prepare_presheath_background(
    plasma: ArgonPresheathPlasmaSI,
    simulation: SimulationConfigSI,
    config: PresheathEquilibrationConfig,
    *,
    collision_model: ArgonNullCollisionModel | None = None,
) -> CylindricalSplineBackground:
    """Equilibrate noninteracting ions and build their compensating potential."""
    plasma.electric_field.validate_for_domain(simulation.height_m)
    collision = collision_model or ArgonNullCollisionModel(plasma)
    collision.validate_timestep(simulation.dt_s)
    softening = simulation.softening_m or 1.0e-3 * plasma.screening_length_m
    fingerprint = _fingerprint(plasma, simulation, config, softening, collision.checksums)
    cache_path = Path(config.output_prefix).expanduser().with_name(
        Path(config.output_prefix).name + "_background.npz"
    )
    if cache_path.is_file():
        cached = load_presheath_background(cache_path)
        if cached.fingerprint == fingerprint:
            return cached
    count = simulation.particle_count
    rng = np.random.default_rng(simulation.seed)
    radius = simulation.radius_m * np.sqrt(rng.random(count))
    theta = 2.0 * np.pi * rng.random(count)
    positions = jnp.asarray(
        np.stack((radius * np.cos(theta), radius * np.sin(theta), simulation.height_m * (rng.random(count) - 0.5)), axis=1),
        jnp.float32,
    )
    velocities_np = rng.normal(0.0, plasma.ion_thermal_speed_m_s, (count, 3))
    velocities_np[:, 2] += plasma.flow_speed_m_s
    velocities = jnp.asarray(velocities_np, jnp.float32)
    key = jax.random.PRNGKey(simulation.seed)
    inlet_values, inlet_cdf = _flux_table(plasma)
    nz_profile = max(8, int(math.ceil(simulation.height_m / (config.grid_spacing_screening_lengths * plasma.screening_length_m))))

    def run_block(carry, block_steps: int):
        def one_step(inner, _):
            pos, vel, random_key, density_sum, flux_sum, out_count = inner
            random_key, collision_key, reinjection_key = jax.random.split(random_key, 3)
            vel, events = collision.apply(collision_key, vel, plasma.ion_thermal_speed_m_s, simulation.dt_s)
            acceleration = ELEMENTARY_CHARGE_C * _field_values(plasma, pos[:, 2]) / plasma.ion_mass_kg
            vel = vel.at[:, 2].add(0.5 * simulation.dt_s * acceleration)
            pos = pos + simulation.dt_s * vel
            acceleration_new = ELEMENTARY_CHARGE_C * _field_values(plasma, pos[:, 2]) / plasma.ion_mass_kg
            vel = vel.at[:, 2].add(0.5 * simulation.dt_s * acceleration_new)

            radial = jnp.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)
            side = radial > simulation.radius_m
            radial_unit = pos[:, :2] / jnp.maximum(radial[:, None], 1.0e-20)
            radial_velocity = jnp.sum(vel[:, :2] * radial_unit, axis=1)
            vel = vel.at[:, :2].add(jnp.where(side[:, None], -2.0 * radial_velocity[:, None] * radial_unit, 0.0))
            reflected_radius = jnp.clip(2.0 * simulation.radius_m - radial, 0.0, simulation.radius_m)
            pos = pos.at[:, :2].set(jnp.where(side[:, None], reflected_radius[:, None] * radial_unit, pos[:, :2]))

            axial_out = (pos[:, 2] < -0.5 * simulation.height_m) | (pos[:, 2] > 0.5 * simulation.height_m)
            keys = jax.random.split(reinjection_key, 4)
            u = jax.random.uniform(keys[0], (count, 3), dtype=jnp.float32)
            normal = jax.random.normal(keys[1], (count, 2), dtype=jnp.float32)
            disk_r = simulation.radius_m * jnp.sqrt(u[:, 0])
            angle = 2.0 * jnp.pi * u[:, 1]
            injected_pos = jnp.stack((disk_r * jnp.cos(angle), disk_r * jnp.sin(angle), jnp.full(count, -0.5 * simulation.height_m)), axis=1)
            injected_vz = jnp.interp(u[:, 2], inlet_cdf, inlet_values)
            injected_vel = jnp.stack((plasma.ion_thermal_speed_m_s * normal[:, 0], plasma.ion_thermal_speed_m_s * normal[:, 1], injected_vz), axis=1)
            pos = jnp.where(axial_out[:, None], injected_pos, pos)
            vel = jnp.where(axial_out[:, None], injected_vel, vel)

            index = jnp.clip(((pos[:, 2] / simulation.height_m + 0.5) * nz_profile).astype(jnp.int32), 0, nz_profile - 1)
            density_sum += jnp.bincount(index, length=nz_profile)
            flux_sum += jnp.bincount(index, weights=vel[:, 2], length=nz_profile)
            out_of_range = jnp.sum(events.out_of_range, dtype=jnp.int32)
            return (pos, vel, random_key, density_sum, flux_sum, out_count + out_of_range), None

        initial = (carry[0], carry[1], carry[2], jnp.zeros(nz_profile), jnp.zeros(nz_profile), jnp.asarray(0, jnp.int32))
        final, _ = jax.lax.scan(one_step, initial, xs=None, length=block_steps)
        return final

    transit_time = simulation.height_m / plasma.flow_speed_m_s
    block_steps = max(1, int(math.ceil(config.block_transit_times * transit_time / simulation.dt_s)))
    minimum_steps = int(math.ceil(config.minimum_transit_times * transit_time / simulation.dt_s))
    maximum_steps = int(math.ceil(config.maximum_transit_times * transit_time / simulation.dt_s))
    compiled_blocks: dict[int, object] = {}

    def execute(steps: int):
        nonlocal positions, velocities, key
        if steps not in compiled_blocks:
            compiled_blocks[steps] = jax.jit(lambda carry, _steps=steps: run_block(carry, _steps))
        final = compiled_blocks[steps]((positions, velocities, key))
        positions, velocities, key = final[0], final[1], final[2]
        if int(np.asarray(jax.device_get(final[5]))) > 0:
            raise RuntimeError("Ar+/Ar collision energy exceeded the 10 keV table range")
        return np.asarray(jax.device_get(final[3])), np.asarray(jax.device_get(final[4]))

    completed = 0
    while completed < minimum_steps:
        steps = min(block_steps, minimum_steps - completed)
        execute(steps)
        completed += steps

    previous_density = previous_flux = None
    consecutive = 0
    history: list[tuple[float, float, float]] = []
    while completed < maximum_steps and consecutive < config.consecutive_blocks:
        steps = min(block_steps, maximum_steps - completed)
        density, flux = execute(steps)
        completed += steps
        density = density / float(steps)
        flux = flux / float(steps)
        if previous_density is None:
            density_change = flux_change = math.inf
        else:
            density_change = _relative_change(density, previous_density)
            flux_change = _relative_change(flux, previous_flux)
            if density_change <= config.density_tolerance and flux_change <= config.flux_tolerance:
                consecutive += 1
            else:
                consecutive = 0
        history.append((completed * simulation.dt_s, density_change, flux_change))
        previous_density, previous_flux = density, flux
    if consecutive < config.consecutive_blocks:
        failure = Path(config.output_prefix).with_name(Path(config.output_prefix).name + "_convergence_failed.json")
        failure.parent.mkdir(parents=True, exist_ok=True)
        failure.write_text(json.dumps({"fingerprint": fingerprint, "history": history}, indent=2), encoding="utf-8")
        raise RuntimeError(f"presheath did not reach steady state; diagnostics: {failure}")

    spacing = config.grid_spacing_screening_lengths * plasma.screening_length_m
    nr = max(4, int(math.ceil(simulation.radius_m / spacing)) + 1)
    nz = max(4, int(math.ceil(simulation.height_m / spacing)) + 1)
    r_axis = np.linspace(0.0, simulation.radius_m, nr)
    z_axis = np.linspace(-0.5 * simulation.height_m, 0.5 * simulation.height_m, nz)
    volume = math.pi * simulation.radius_m**2 * simulation.height_m
    superion_weight = plasma.reference_density_m3 * volume / count
    superion_charge = ELEMENTARY_CHARGE_C * superion_weight
    density_accum = np.zeros((nz, nr), dtype=np.float64)
    axial_flux_accum = np.zeros((nz, nr), dtype=np.float64)
    potential_accum = np.zeros((nz, nr), dtype=np.float64)
    for _ in range(config.averaging_blocks):
        execute(block_steps)
        host_positions = np.asarray(jax.device_get(positions))
        host_velocities = np.asarray(jax.device_get(velocities))
        density_accum += _axisymmetric_node_density(
            host_positions,
            r_axis,
            z_axis,
            superion_weight,
            simulation.radius_m,
            simulation.height_m,
        )
        axial_flux_accum += _axisymmetric_node_density(
            host_positions,
            r_axis,
            z_axis,
            superion_weight,
            simulation.radius_m,
            simulation.height_m,
            particle_values=host_velocities[:, 2],
        )
        potential_accum += _direct_potential(host_positions, r_axis, z_axis, superion_charge, plasma.screening_length_m, softening, config.source_block_size)
    density_mean = density_accum / config.averaging_blocks
    axial_flux_mean = axial_flux_accum / config.averaging_blocks
    ion_potential = potential_accum / config.averaging_blocks
    metadata = {
        "plasma": asdict(plasma),
        "simulation": asdict(simulation),
        "equilibration": {**asdict(config), "output_prefix": str(config.output_prefix)},
        "softening_m": softening,
        "cross_section_sha256": collision.checksums,
        "completed_steps": completed + config.averaging_blocks * block_steps,
        "boundary_model": "upstream_reinjection_specular_side",
    }
    background = CylindricalSplineBackground(
        r_m=r_axis,
        z_m=z_axis,
        density_m3=density_mean,
        ion_potential_V=ion_potential,
        potential_V=-ion_potential,
        fingerprint=fingerprint,
        convergence_history=np.asarray(history, dtype=np.float64),
        metadata=metadata,
        axial_flux_m2_s=axial_flux_mean,
    )
    save_presheath_background(background, config.output_prefix)
    return background


__all__ = [
    "CylindricalSplineBackground",
    "cylindrical_cubic_value_and_gradient",
    "load_presheath_background",
    "prepare_presheath_background",
    "save_presheath_background",
]
