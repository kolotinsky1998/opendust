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
    presheath_preparation_speed_limit_m_s,
)
from opendust_jax.models import (
    COULOMB_CONSTANT,
    ArgonPresheathPlasmaSI,
    CollisionSubcyclingConfig,
    PolynomialBackgroundConfig,
    PresheathEquilibrationConfig,
    QuadraticElectricFieldSI,
    SimulationConfigSI,
)
from opendust_jax.presheath import (
    CylindricalSplineBackground,
    _axisymmetric_node_density,
    _direct_potential,
    _profile_sampling_schedule,
    cylindrical_cubic_value_and_gradient,
    load_presheath_background,
    prepare_presheath_background,
    save_presheath_background,
)
from opendust_jax.simulation import OpenDustJAX
from opendust_jax.timestepping import resolve_collision_substeps


def _cylindrical_node_volumes(r, z, radius, height):
    r_edges = np.concatenate(([0.0], 0.5 * (r[1:] + r[:-1]), [radius]))
    z_edges = np.concatenate(([-0.5 * height], 0.5 * (z[1:] + z[:-1]), [0.5 * height]))
    return np.pi * np.diff(r_edges**2)[None, :] * np.diff(z_edges)[:, None]


def argon_plasma(pressure=0.1, inlet_mach=0.3):
    return ArgonPresheathPlasmaSI(
        neutral_temperature_K=300.0,
        electron_temperature_K=30_000.0,
        reference_density_m3=1.0e14,
        pressure_Pa=pressure,
        inlet_mach=inlet_mach,
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


def test_argon_collision_substeps_use_null_majorant_limit():
    plasma = argon_plasma()
    model = ArgonNullCollisionModel(plasma)
    limit = -math.log(0.9) / model.maximum_frequency_s
    resolved, dt_collision, frequency, returned_limit = resolve_collision_substeps(
        plasma,
        2.5 * limit,
        CollisionSubcyclingConfig(),
        model,
    )
    assert resolved == 3
    assert dt_collision <= limit
    assert frequency == pytest.approx(model.maximum_frequency_s)
    assert returned_limit == pytest.approx(limit)


def test_bounded_preparation_majorant_uses_two_sound_plus_six_thermal_speeds():
    plasma = ArgonPresheathPlasmaSI(
        neutral_temperature_K=290.0,
        electron_temperature_K=58_022.0,
        reference_density_m3=1.0e14,
        pressure_Pa=19.9984,
        inlet_mach=0.3,
        electric_field=QuadraticElectricFieldSI(
            1000.0, enforce_presheath_constraints=False
        ),
    )
    speed_limit = presheath_preparation_speed_limit_m_s(plasma)
    model = ArgonNullCollisionModel(
        plasma, maximum_relative_speed_m_s=speed_limit
    )
    assert speed_limit == pytest.approx(8424.253976610902)
    assert model.maximum_majorant_energy_eV == pytest.approx(7.345758053432533)
    assert model.maximum_frequency_s == pytest.approx(2.0140937e7, rel=2e-6)
    limit = -math.log(0.9) / model.maximum_frequency_s
    assert limit == pytest.approx(5.2311625e-9, rel=2e-6)
    full = ArgonNullCollisionModel(plasma)
    assert full.maximum_relative_speed_m_s is None
    assert full.maximum_majorant_energy_eV == pytest.approx(10_000.0)
    assert full.maximum_frequency_s > 10.0 * model.maximum_frequency_s


def test_bounded_majorant_saturates_and_reports_faster_particles():
    model = ArgonNullCollisionModel(
        argon_plasma(), maximum_relative_speed_m_s=500.0
    )
    dt = -math.log(0.9) / model.maximum_frequency_s
    velocities = jnp.tile(jnp.asarray([[3000.0, 0.0, 0.0]], jnp.float32), (4096, 1))
    _, events = model.apply(jax.random.PRNGKey(27), velocities, 1.0, dt)
    assert np.mean(np.asarray(events.majorant_exceeded)) > 0.99
    assert 0.07 < np.mean(np.asarray(events.candidates)) < 0.13


def test_sampling_window_expands_to_plasma_period_when_transit_is_short(tmp_path):
    plasma = argon_plasma(pressure=0.1)
    simulation = SimulationConfigSI(
        radius_m=0.001,
        height_m=0.001,
        particle_count=8,
        steps=0,
        dt_s=1.0e-9,
        force_backend="jax",
    )
    config = PresheathEquilibrationConfig(
        output_prefix=tmp_path / "sampling",
        minimum_transit_times=0.1,
        maximum_transit_times=10.0,
    )
    (
        period,
        _,
        interval_steps,
        resolved_interval,
        window_steps,
        window,
    ) = _profile_sampling_schedule(plasma, simulation, config)
    transit = simulation.height_m / plasma.flow_speed_m_s
    assert transit < period
    assert resolved_interval >= period
    assert window >= resolved_interval
    assert window_steps % interval_steps == 0


@pytest.mark.parametrize("inlet_mach", [0.46, 0.92, 1.40])
def test_article_sampling_window_has_at_least_eighteen_profiles(
    tmp_path, inlet_mach
):
    plasma = argon_plasma(inlet_mach=inlet_mach)
    simulation = SimulationConfigSI(
        radius_m=1.25 * plasma.screening_length_m,
        height_m=5.0 * plasma.screening_length_m,
        particle_count=8,
        steps=0,
        dt_s=5.231162530152083e-9,
        force_backend="jax",
    )
    config = PresheathEquilibrationConfig(
        output_prefix=tmp_path / f"sampling_{inlet_mach}",
        minimum_transit_times=4.0,
        block_transit_times=10.0,
        maximum_transit_times=140.0,
        consecutive_blocks=2,
        minimum_samples_per_convergence_window=18,
    )
    (
        _,
        _,
        interval_steps,
        _,
        window_steps,
        window_s,
    ) = _profile_sampling_schedule(plasma, simulation, config)
    transit_s = simulation.height_m / plasma.flow_speed_m_s
    samples_per_window = window_steps // interval_steps
    maximum_steps = math.ceil(
        config.maximum_transit_times * transit_s / simulation.dt_s
    )
    minimum_steps = math.ceil(
        config.minimum_transit_times * transit_s / simulation.dt_s
    )

    assert samples_per_window == 18
    assert window_s >= config.block_transit_times * transit_s
    assert (maximum_steps - minimum_steps) // window_steps >= (
        config.consecutive_blocks + 1
    )


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
    assert restored._device_r_m is not None
    assert restored._device_z_m is not None
    assert restored._device_potential_V is not None
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
    density = np.asarray(density)
    flux = np.asarray(flux)
    assert np.sum(density * volumes) == pytest.approx(
        len(positions) * weight, rel=2e-6
    )
    assert np.sum(flux * volumes) == pytest.approx(
        np.sum(velocities) * weight, rel=2e-6
    )
    velocity = np.divide(flux, density, out=np.zeros_like(flux), where=density > 0)
    expected = np.divide(flux, density, out=np.zeros_like(flux), where=density > 0)
    np.testing.assert_allclose(velocity[density > 0], expected[density > 0])


def test_presheath_direct_potential_matches_numpy_reference():
    positions = np.asarray(
        [[0.1, -0.2, 0.3], [0.4, 0.2, -0.1], [-0.3, 0.1, 0.2]],
        dtype=np.float32,
    )
    targets = np.asarray(
        [[0.0, 0.0, -0.5], [0.2, 0.0, 0.0], [0.0, 0.0, 0.5]],
        dtype=np.float32,
    )
    charge, screening, softening = 2.0e-16, 0.8, 0.01
    distance = np.linalg.norm(
        targets[:, None, :].astype(np.float64)
        - positions[None, :, :].astype(np.float64),
        axis=-1,
    )
    expected = (
        COULOMB_CONSTANT
        * charge
        * np.sum(np.exp(-distance / screening) / (distance + softening), axis=1)
    ).reshape(3, 1)
    actual = _direct_potential(
        positions, targets, (3, 1), charge, screening, softening, block_size=2
    )
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=3e-6, atol=1e-12)


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


def test_prepare_then_run_small_presheath_case(tmp_path, monkeypatch):
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
    collision_subcycling = CollisionSubcyclingConfig(substeps=2)
    preparation = SimulationConfigSI(
        steps=0,
        collision_subcycling=collision_subcycling,
        **common,
    )
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
        sampling_interval_plasma_periods=0.001,
    )
    transferred_shapes = []
    real_device_get = jax.device_get

    def recording_device_get(value):
        transferred_shapes.extend(
            getattr(leaf, "shape", None) for leaf in jax.tree_util.tree_leaves(value)
        )
        return real_device_get(value)

    monkeypatch.setattr(jax, "device_get", recording_device_get)
    background = prepare_presheath_background(plasma, preparation, equilibration)
    assert background.density_m3.shape == background.potential_V.shape
    assert np.all(np.isfinite(background.potential_V))
    assert (common["particle_count"], 3) not in transferred_shapes
    assert background._device_r_m is not None
    assert background._device_z_m is not None
    assert background._device_potential_V is not None
    assert {
        device.platform for device in background._device_potential_V.devices()
    } == {jax.default_backend()}
    cached = prepare_presheath_background(plasma, preparation, equilibration)
    assert cached.fingerprint == background.fingerprint
    np.testing.assert_array_equal(cached.potential_V, background.potential_V)
    assert background.metadata["collision_subcycling"]["resolved_substeps"] == 2
    assert background.metadata["collision_subcycling"]["dt_collision_s"] == pytest.approx(
        common["dt_s"] / 2
    )
    sampling = background.metadata["profile_sampling"]
    assert sampling["resolved_interval_steps"] >= 1
    assert sampling["convergence_window_s"] >= sampling["resolved_interval_s"]
    assert sampling["samples_per_window"] == (
        sampling["convergence_window_steps"]
        // sampling["resolved_interval_steps"]
    )
    assert sampling["convergence_sample_count"] < background.metadata["completed_steps"]
    production = SimulationConfigSI(
        steps=1,
        collision_subcycling=collision_subcycling,
        **common,
    )
    runner = OpenDustJAX(plasma, production, background=background)
    assert runner.collision_substeps == 2
    result = runner.run()
    assert int(result.final_state.step) == 1
    assert result.collision_statistics.operator_call_count == 2

    different_schedule = SimulationConfigSI(steps=1, dt_s=2.0e-8, **{
        key: value for key, value in common.items() if key != "dt_s"
    })
    OpenDustJAX(plasma, different_schedule, background=background)
