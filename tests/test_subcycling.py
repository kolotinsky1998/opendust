import math
import sys
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from opendust_jax import (
    ChargingSubcyclingConfig,
    DustParticleSI,
    FieldDrivenPlasmaSI,
    MaxwellianPlasmaSI,
    OpenDustJAX,
    OutputConfig,
    PolynomialBackgroundConfig,
    SimulationConfigSI,
    SimulationState,
    recommend_main_timestep,
)
from opendust_jax.physics import update_dust_charges_C


ION_MASS = 6.6335209e-26


def maxwellian() -> MaxwellianPlasmaSI:
    return MaxwellianPlasmaSI(300.0, 30_000.0, 1.0e3, 0.5, ION_MASS)


def simulation(**overrides) -> SimulationConfigSI:
    values = dict(
        radius_m=1.0,
        height_m=2.0,
        particle_count=2,
        steps=1,
        dt_s=1.0e-5,
        seed=7,
        force_backend="jax",
        step_chunk_size=1,
    )
    values.update(overrides)
    return SimulationConfigSI(**values)


def engine(plasma, config, dust=(), output=None) -> OpenDustJAX:
    return OpenDustJAX(
        plasma,
        config,
        dust,
        background=PolynomialBackgroundConfig(enabled=False),
        output=output,
    )


def test_subcycling_configuration_validation():
    with pytest.raises(ValueError, match="substeps"):
        ChargingSubcyclingConfig(substeps=0)
    with pytest.raises(ValueError, match="max_displacement_fraction"):
        ChargingSubcyclingConfig(max_displacement_fraction=0.0)
    with pytest.raises(ValueError, match="velocity_limit_m_s"):
        ChargingSubcyclingConfig(velocity_limit_m_s=-1.0)


def test_automatic_substeps_use_smallest_self_consistent_dust():
    plasma = maxwellian()
    config = ChargingSubcyclingConfig(
        max_displacement_fraction=0.1,
        thermal_speed_multiplier=4.0,
    )
    dust = (
        DustParticleSI((0.0, 0.0, -0.3), 0.02, charge_mode="self_consistent"),
        DustParticleSI((0.0, 0.0, 0.3), 0.01, charge_mode="self_consistent"),
    )
    sim = simulation(dt_s=2.0e-6, charging_subcycling=config)
    actual = engine(plasma, sim, dust).charging_substeps
    speed = abs(plasma.flow_speed_m_s) + 4.0 * plasma.ion_thermal_speed_m_s
    expected = max(1, math.ceil(sim.dt_s * speed / (0.1 * 0.01)))
    assert actual == expected


def test_velocity_limit_override_and_no_consistent_dust():
    dust = DustParticleSI((0.0, 0.0, 0.0), 0.01, charge_mode="fixed")
    automatic = engine(
        maxwellian(),
        simulation(
            dt_s=1.0e-3,
            charging_subcycling=ChargingSubcyclingConfig(velocity_limit_m_s=100.0),
        ),
        [dust],
    )
    assert automatic.charging_substeps == 1
    explicit = engine(
        maxwellian(),
        simulation(charging_subcycling=ChargingSubcyclingConfig(substeps=7)),
        [dust],
    )
    assert explicit.charging_substeps == 7
    assert explicit.charging_dt_s == pytest.approx(explicit.simulation.dt_s / 7)


def test_main_timestep_recommendation():
    plasma = FieldDrivenPlasmaSI(300.0, 30_000.0, 1.0e10, 2.0, 5.0e4, ION_MASS)
    recommendation = recommend_main_timestep(plasma)
    expected_omega = math.sqrt(
        plasma.density_m3 * 1.602176634e-19**2
        / (8.8541878128e-12 * plasma.ion_mass_kg)
    )
    assert recommendation.ion_plasma_frequency_s == pytest.approx(expected_omega)
    assert recommendation.collision_limit_s == pytest.approx(0.1 / 5.0e4)
    assert recommendation.recommended_dt_s == pytest.approx(
        min(0.1 / expected_omega, 0.1 / 5.0e4)
    )


def test_subcycling_catches_ion_that_crosses_dust_between_main_endpoints():
    plasma = MaxwellianPlasmaSI(1.0, 1.0, 1.0, 0.0, ION_MASS)
    dust = DustParticleSI((0.0, 0.0, 0.0), 5.0e-4, charge_mode="fixed")
    common = dict(
        radius_m=0.01,
        height_m=0.02,
        particle_count=1,
        steps=1,
        dt_s=0.1,
        seed=2,
        force_backend="jax",
        step_chunk_size=1,
    )
    state = SimulationState(
        positions_m=jnp.asarray([[-2.0e-3, 0.0, 0.0]], jnp.float32),
        velocities_m_s=jnp.asarray([[4.0e-2, 0.0, 0.0]], jnp.float32),
        forces_N=jnp.zeros((1, 3), jnp.float32),
        dust_charges_C=jnp.asarray([0.0], jnp.float32),
        step=jnp.asarray(0, jnp.int32),
        prng_key=jax.random.PRNGKey(9),
    )
    without = engine(plasma, SimulationConfigSI(**common), [dust]).run(state)
    with_substeps = engine(
        plasma,
        SimulationConfigSI(
            **common,
            charging_subcycling=ChargingSubcyclingConfig(substeps=10),
        ),
        [dust],
    ).run(state)
    assert np.all(without.dust_collection_forces_N[1, 0] == 0.0)
    assert np.any(with_substeps.dust_collection_forces_N[1, 0] != 0.0)


def test_collisions_are_sampled_once_per_main_step():
    plasma = FieldDrivenPlasmaSI(300.0, 30_000.0, 1.0, 0.0, 10.0, ION_MASS)
    sim = simulation(
        particle_count=8,
        steps=2,
        dt_s=0.1,
        charging_subcycling=ChargingSubcyclingConfig(substeps=5),
    )
    with pytest.warns(RuntimeWarning, match="exceeds 0.2"):
        result = engine(plasma, sim).run()
    assert result.collision_statistics.candidate_count == 16
    assert result.collision_statistics.charge_exchange_count == 16


def test_self_consistent_electron_current_is_integrated_each_substep():
    plasma = maxwellian()
    dust = DustParticleSI(
        (0.0, 0.0, 0.0),
        0.01,
        initial_charge_C=-1.0e-18,
        charge_mode="self_consistent",
    )
    sim = simulation(
        particle_count=1,
        dt_s=1.0e-8,
        charging_subcycling=ChargingSubcyclingConfig(substeps=4),
    )
    runner = engine(plasma, sim, [dust])
    state = SimulationState(
        positions_m=jnp.asarray([[0.5, 0.0, 0.0]], jnp.float32),
        velocities_m_s=jnp.zeros((1, 3), jnp.float32),
        forces_N=jnp.zeros((1, 3), jnp.float32),
        dust_charges_C=jnp.asarray([dust.initial_charge_C], jnp.float32),
        step=jnp.asarray(0, jnp.int32),
        prng_key=jax.random.PRNGKey(4),
    )
    result = runner.run(state)
    expected = state.dust_charges_C
    for _ in range(4):
        expected = update_dust_charges_C(
            expected,
            jnp.asarray([2], jnp.int32),
            jnp.zeros((1,), jnp.float32),
            jnp.asarray([dust.radius_m], jnp.float32),
            runner.superion_charge_C,
            plasma.density_m3,
            plasma.electron_temperature_K,
            plasma.ion_temperature_K,
            plasma.ion_mass_kg,
            plasma.flow_speed_m_s,
            sim.dt_s / 4,
        )
    np.testing.assert_allclose(result.dust_charges_C[-1], np.asarray(expected), rtol=1e-6)


def test_subcycling_is_independent_of_jit_chunk_size():
    config = ChargingSubcyclingConfig(substeps=3)
    left = engine(
        maxwellian(),
        simulation(steps=4, step_chunk_size=1, charging_subcycling=config),
    ).run()
    right = engine(
        maxwellian(),
        simulation(steps=4, step_chunk_size=4, charging_subcycling=config),
    ).run()
    np.testing.assert_array_equal(
        np.asarray(left.final_state.positions_m),
        np.asarray(right.final_state.positions_m),
    )
    np.testing.assert_array_equal(
        np.asarray(left.final_state.velocities_m_s),
        np.asarray(right.final_state.velocities_m_s),
    )


def test_one_substep_matches_legacy_integrator_without_reinjection():
    initial = engine(maxwellian(), simulation(steps=0)).initial_state()
    legacy = engine(maxwellian(), simulation(steps=2)).run(initial)
    respa = engine(
        maxwellian(),
        simulation(
            steps=2,
            charging_subcycling=ChargingSubcyclingConfig(substeps=1),
        ),
    ).run(initial)
    np.testing.assert_allclose(
        np.asarray(respa.final_state.positions_m),
        np.asarray(legacy.final_state.positions_m),
        rtol=2.0e-6,
        atol=1.0e-7,
    )
    np.testing.assert_allclose(
        np.asarray(respa.final_state.velocities_m_s),
        np.asarray(legacy.final_state.velocities_m_s),
        rtol=2.0e-6,
        atol=1.0e-7,
    )


def test_checkpoint_requires_matching_subcycling(tmp_path):
    checkpoint = tmp_path / "restart.npz"
    config = ChargingSubcyclingConfig(substeps=2)
    continuous = engine(
        maxwellian(), simulation(steps=4, charging_subcycling=config)
    ).run()
    producer = engine(
        maxwellian(),
        simulation(steps=2, charging_subcycling=config),
        output=OutputConfig(checkpoint_path=checkpoint),
    )
    producer.run()
    matching = engine(
        maxwellian(), simulation(steps=2, charging_subcycling=config)
    ).run_from_checkpoint(checkpoint)
    assert int(matching.final_state.step) == 4
    np.testing.assert_array_equal(
        np.asarray(matching.final_state.positions_m),
        np.asarray(continuous.final_state.positions_m),
    )
    np.testing.assert_array_equal(
        np.asarray(matching.final_state.velocities_m_s),
        np.asarray(continuous.final_state.velocities_m_s),
    )
    mismatching = engine(maxwellian(), simulation())
    with pytest.raises(ValueError, match="subcycling configuration"):
        mismatching.run_from_checkpoint(checkpoint)
