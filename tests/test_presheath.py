import math
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from opendust_jax.collisions import (
    ArgonNullCollisionModel,
    elastic_scatter_equal_mass,
    load_argon_cross_sections,
)
from opendust_jax.models import (
    ArgonPresheathPlasmaSI,
    PolynomialBackgroundConfig,
    PresheathEquilibrationConfig,
    QuadraticElectricFieldSI,
    SimulationConfigSI,
)
from opendust_jax.simulation import OpenDustJAX
from opendust_jax.presheath import (
    CylindricalSplineBackground,
    _axisymmetric_node_density,
    cylindrical_cubic_value_and_gradient,
    load_presheath_background,
    save_presheath_background,
    prepare_presheath_background,
)


def _cylindrical_node_volumes(r, z, radius, height):
    r_edges = np.concatenate(([0.0], 0.5 * (r[1:] + r[:-1]), [radius]))
    z_edges = np.concatenate(([-0.5 * height], 0.5 * (z[1:] + z[:-1]), [0.5 * height]))
    return np.pi * np.diff(r_edges**2)[None, :] * np.diff(z_edges)[:, None]


def argon_plasma(pressure=0.1):
    return ArgonPresheathPlasmaSI(
        neutral_temperature_K=300.0,
        electron_temperature_K=30_000.0,
        reference_density_m3=1.0e14,
        pressure_Pa=pressure,
        inlet_mach=0.3,
        electric_field=QuadraticElectricFieldSI(
            1000.0, enforce_presheath_constraints=False
        ),
    )


def test_argon_tables_are_validated_and_share_energy_grid():
    data = load_argon_cross_sections(REPO_ROOT / "data" / "cross_sections")
    assert data.energy_frame == "center_of_mass"
    assert data.maximum_energy_eV == 10_000.0
    assert len(data.energy_eV) == 114
    assert data.checksums[0].startswith("64845aed")
    assert data.checksums[1].startswith("e095b982")


def test_log_interpolation_reproduces_table_knots():
    data = load_argon_cross_sections(REPO_ROOT / "data" / "cross_sections")
    model = ArgonNullCollisionModel(argon_plasma(), data)
    elastic, cx = model.cross_sections_m2(jnp.asarray(data.energy_eV, jnp.float32))
    np.testing.assert_allclose(elastic, data.elastic_m2, rtol=2e-5)
    np.testing.assert_allclose(cx, data.charge_exchange_m2, rtol=2e-5)


def test_equal_mass_elastic_scattering_conserves_pair_momentum_and_energy():
    ion = jnp.asarray([[4.0, -2.0, 1.0], [1.0, 5.0, -3.0]], jnp.float32)
    neutral = jnp.asarray([[-1.0, 3.0, 2.0], [2.0, -4.0, 0.5]], jnp.float32)
    direction = jnp.asarray([[0.2, 0.4, 0.8], [-0.3, 0.7, 0.1]], jnp.float32)
    ion_after, neutral_after = elastic_scatter_equal_mass(ion, neutral, direction)
    np.testing.assert_allclose(ion_after + neutral_after, ion + neutral, rtol=2e-6)
    before = np.sum(np.asarray(ion) ** 2 + np.asarray(neutral) ** 2, axis=1)
    after = np.sum(np.asarray(ion_after) ** 2 + np.asarray(neutral_after) ** 2, axis=1)
    np.testing.assert_allclose(after, before, rtol=2e-6)


def test_null_collision_channels_partition_candidates():
    model = ArgonNullCollisionModel(argon_plasma())
    dt = -math.log(0.95) / model.maximum_frequency_s
    model.validate_timestep(dt)
    velocities = jnp.tile(jnp.asarray([[2000.0, 0.0, 0.0]], jnp.float32), (50_000, 1))
    _, events = model.apply(jax.random.PRNGKey(4), velocities, 250.0, dt)
    candidate = np.asarray(events.candidates)
    elastic = np.asarray(events.elastic)
    cx = np.asarray(events.charge_exchange)
    null = np.asarray(events.null)
    assert 0.04 < candidate.mean() < 0.06
    np.testing.assert_array_equal(elastic | cx | null, candidate)
    assert not np.any(elastic & cx)
    assert not np.any((elastic | cx) & null)


def test_candidate_probability_guard_rejects_large_timestep():
    model = ArgonNullCollisionModel(argon_plasma())
    with pytest.raises(ValueError, match="reduce dt_s"):
        model.validate_timestep(1.0 / model.maximum_frequency_s)


def test_collision_operator_flags_energy_above_table_range():
    model = ArgonNullCollisionModel(argon_plasma())
    dt = -math.log(0.90) / model.maximum_frequency_s
    velocities = jnp.tile(jnp.asarray([[1.0e7, 0.0, 0.0]], jnp.float32), (2000, 1))
    _, events = model.apply(jax.random.PRNGKey(19), velocities, 250.0, dt)
    assert int(np.sum(np.asarray(events.out_of_range))) > 100


def test_quadratic_field_clamps_and_validates_shape():
    field = QuadraticElectricFieldSI(
        E0_V_m=10.0,
        alpha_V_m2=1.0,
        beta_V_m3=-1.0,
        z_min_m=-1.0,
        z_max_m=1.0,
    )
    field.validate_for_domain(2.0, lowest_dust_z_m=0.0)
    assert field.value_V_m(-2.0) == field.value_V_m(-1.0)
    assert field.value_V_m(2.0) == field.value_V_m(1.0)
    with pytest.raises(ValueError, match="beta"):
        QuadraticElectricFieldSI(1.0, 0.0, 1.0).validate_for_domain(2.0)


def test_cylindrical_cubic_gradient_matches_finite_difference():
    r = jnp.linspace(0.0, 2.0, 17)
    z = jnp.linspace(-1.0, 1.0, 17)
    rr, zz = jnp.meshgrid(r, z, indexing="xy")
    values = 0.3 * rr**2 + 0.2 * zz**2 + 0.1 * rr * zz
    position = jnp.asarray([[0.8, 0.6, 0.25]], jnp.float32)
    value, gradient = cylindrical_cubic_value_and_gradient(position, r, z, values)
    h = 2.0e-4
    numerical = []
    for axis in range(3):
        plus, minus = np.asarray(position).copy(), np.asarray(position).copy()
        plus[0, axis] += h
        minus[0, axis] -= h
        vp, _ = cylindrical_cubic_value_and_gradient(plus, r, z, values)
        vm, _ = cylindrical_cubic_value_and_gradient(minus, r, z, values)
        numerical.append(float((vp[0] - vm[0]) / (2 * h)))
    np.testing.assert_allclose(np.asarray(gradient[0]), numerical, rtol=4e-3, atol=4e-4)
    assert np.isfinite(float(value[0]))
    _, on_axis = cylindrical_cubic_value_and_gradient(
        jnp.asarray([[0.0, 0.0, 0.25]], jnp.float32), r, z, values
    )
    np.testing.assert_array_equal(np.asarray(on_axis[0, :2]), [0.0, 0.0])


def test_presheath_background_round_trip(tmp_path):
    r = np.linspace(0.0, 1.0, 4)
    z = np.linspace(-1.0, 1.0, 5)
    shape = (len(z), len(r))
    axial_flux = np.arange(np.prod(shape), dtype=float).reshape(shape)
    background = CylindricalSplineBackground(
        r, z, np.ones(shape), np.full(shape, 2.0), np.full(shape, -2.0),
        "abc123", np.asarray([[1.0, 0.01, 0.02]]), {"case": "test"}, axial_flux,
    )
    npz, json_path = save_presheath_background(background, tmp_path / "case")
    restored = load_presheath_background(npz)
    np.testing.assert_array_equal(restored.potential_V, background.potential_V)
    assert restored.fingerprint == "abc123"
    assert restored.metadata == {"case": "test"}
    np.testing.assert_array_equal(restored.axial_flux_m2_s, axial_flux)
    np.testing.assert_array_equal(restored.axial_velocity_m_s, axial_flux)
    assert json_path.is_file()


def test_axisymmetric_density_and_flux_preserve_particle_totals():
    radius, height = 1.0, 2.0
    r, z = np.linspace(0.0, radius, 6), np.linspace(-1.0, 1.0, 7)
    positions = np.asarray([[0.1, 0.0, -0.7], [0.3, 0.4, 0.1], [0.8, 0.0, 0.8]])
    velocities = np.asarray([2.0, 3.0, 5.0])
    weight = 7.0
    density = _axisymmetric_node_density(positions, r, z, weight, radius, height)
    flux = _axisymmetric_node_density(
        positions, r, z, weight, radius, height, particle_values=velocities
    )
    volumes = _cylindrical_node_volumes(r, z, radius, height)
    assert np.sum(density * volumes) == pytest.approx(len(positions) * weight)
    assert np.sum(flux * volumes) == pytest.approx(np.sum(velocities) * weight)
    velocity = np.divide(flux, density, out=np.zeros_like(flux), where=density > 0)
    expected = np.divide(flux, density, out=np.zeros_like(flux), where=density > 0)
    np.testing.assert_allclose(velocity[density > 0], expected[density > 0])


def test_presheath_production_path_runs_with_matching_background():
    plasma = argon_plasma(pressure=1.0e-4)
    simulation = SimulationConfigSI(
        radius_m=0.01,
        height_m=0.02,
        particle_count=8,
        steps=2,
        dt_s=1.0e-9,
        seed=11,
        force_backend="jax",
        step_chunk_size=1,
    )
    collision = ArgonNullCollisionModel(plasma)
    r = np.linspace(0.0, simulation.radius_m, 4)
    z = np.linspace(-0.5 * simulation.height_m, 0.5 * simulation.height_m, 5)
    zeros = np.zeros((len(z), len(r)))
    metadata = {
        "plasma": asdict(plasma),
        "simulation": asdict(simulation),
        "softening_m": 1.0e-3 * plasma.screening_length_m,
        "cross_section_sha256": collision.checksums,
    }
    background = CylindricalSplineBackground(
        r, z, zeros, zeros, zeros, "matching", np.zeros((0, 3)), metadata
    )
    result = OpenDustJAX(
        plasma, simulation, background=background, collision_model=collision
    ).run()
    assert int(result.final_state.step) == 2
    assert result.collision_statistics.out_of_range_count == 0

    with pytest.raises(ValueError, match="requires"):
        OpenDustJAX(
            argon_plasma(),
            simulation,
            background=PolynomialBackgroundConfig(enabled=False),
        )


def test_prepare_then_run_small_presheath_case(tmp_path):
    plasma = ArgonPresheathPlasmaSI(
        300.0,
        30_000.0,
        1.0e12,
        1.0e-4,
        0.3,
        QuadraticElectricFieldSI(100.0, enforce_presheath_constraints=False),
    )
    common = dict(
        radius_m=0.002,
        height_m=0.004,
        particle_count=32,
        dt_s=1.0e-8,
        seed=2,
        force_backend="jax",
        step_chunk_size=2,
    )
    preparation = SimulationConfigSI(steps=0, **common)
    equilibration = PresheathEquilibrationConfig(
        output_prefix=tmp_path / "small",
        grid_spacing_screening_lengths=0.5,
        density_tolerance=1.0e9,
        flux_tolerance=1.0e9,
        consecutive_blocks=1,
        minimum_transit_times=0.05,
        block_transit_times=0.05,
        maximum_transit_times=0.5,
        averaging_blocks=1,
        source_block_size=16,
    )
    background = prepare_presheath_background(plasma, preparation, equilibration)
    assert background.density_m3.shape == background.potential_V.shape
    assert np.all(np.isfinite(background.potential_V))
    cached = prepare_presheath_background(plasma, preparation, equilibration)
    assert cached.fingerprint == background.fingerprint
    np.testing.assert_array_equal(cached.potential_V, background.potential_V)
    production = SimulationConfigSI(steps=1, **common)
    result = OpenDustJAX(plasma, production, background=background).run()
    assert int(result.final_state.step) == 1
