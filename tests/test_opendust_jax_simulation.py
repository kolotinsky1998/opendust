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

from opendust_jax.background import build_polynomial_background
from opendust_jax.checkpoint import load_checkpoint, save_checkpoint
from opendust_jax.collisions import LegacyChargeExchangeCollision
from opendust_jax.diagnostics import plane_cic_density_snapshot_m3
from opendust_jax.models import (
    DustParticleSI,
    FieldDrivenPlasmaSI,
    MaxwellianPlasmaSI,
    PlaneDiagnosticConfig,
    PolynomialBackgroundConfig,
    SimulationConfigSI,
    SimulationState,
    validate_system,
)
from opendust_jax.physics import (
    COULOMB_CONSTANT,
    dust_orbit_forces_N,
    ion_dust_coulomb_forces_N,
    shifted_oml_ion_flux_m2_s,
    polynomial_value_and_gradient,
)
from opendust_jax.simulation import OpenDustJAX
from opendust_jax.pair_forces import softened_yukawa_forces


def plasma():
    return MaxwellianPlasmaSI(
        ion_temperature_K=300.0,
        electron_temperature_K=30_000.0,
        density_m3=1.0e5,
        mach=0.0,
        ion_mass_kg=6.6335209e-26,
    )


def test_piel_asymmetry_uses_bare_ion_force_and_screened_dust_force():
    ion = jnp.asarray([[1.0, 0.0, 0.0]], jnp.float32)
    dust = jnp.asarray([[0.0, 0.0, 0.0]], jnp.float32)
    q_i = 2.0
    q_d = jnp.asarray([-3.0], jnp.float32)
    ion_force = np.asarray(ion_dust_coulomb_forces_N(ion, q_i, dust, q_d))[0]
    dust_force = np.asarray(dust_orbit_forces_N(ion, q_i, dust, q_d, 2.0))[0]
    expected_ion_x = COULOMB_CONSTANT * q_i * float(q_d[0])
    np.testing.assert_allclose(ion_force[0], expected_ion_x, rtol=2e-6)
    expected_dust_x = -expected_ion_x * np.exp(-0.5) * (1.0 + 0.5)
    np.testing.assert_allclose(dust_force[0], expected_dust_x, rtol=2e-6)
    assert not np.isclose(dust_force[0], -ion_force[0])


def test_shifted_oml_zero_drift_has_finite_analytic_limit():
    charge = jnp.asarray([-2.0e-16], jnp.float32)
    radius = jnp.asarray([1.0e-5], jnp.float32)
    density = 1.0e14
    temperature = 300.0
    mass = 6.6335209e-26
    actual = np.asarray(
        shifted_oml_ion_flux_m2_s(
            charge, radius, density, temperature, mass, 0.0
        )
    )
    v_ti = np.sqrt(2.0 * 1.380649e-23 * temperature / mass)
    potential = COULOMB_CONSTANT * np.asarray(charge) / np.asarray(radius)
    chi = -1.602176634e-19 * potential / (1.380649e-23 * temperature)
    expected = density * v_ti / (2.0 * np.sqrt(np.pi)) * (1.0 + chi)
    np.testing.assert_allclose(actual, expected, rtol=2e-6)
    assert np.all(np.isfinite(actual))


def test_shifted_oml_finite_drift_matches_corrected_formula():
    charge = jnp.asarray([-2.0e-16], jnp.float32)
    radius = jnp.asarray([1.0e-5], jnp.float32)
    density, temperature, mass, flow = 1.0e14, 300.0, 6.6335209e-26, 500.0
    actual = np.asarray(
        shifted_oml_ion_flux_m2_s(
            charge, radius, density, temperature, mass, flow
        )
    )
    v_ti = np.sqrt(2 * 1.380649e-23 * temperature / mass)
    u = abs(flow) / v_ti
    potential = COULOMB_CONSTANT * np.asarray(charge) / np.asarray(radius)
    chi = -1.602176634e-19 * potential / (1.380649e-23 * temperature)
    bracket = (
        (1 + (0.5 + chi) / u**2) * special_erf(u)
        + np.exp(-u**2) / (u * np.sqrt(np.pi))
    )
    expected = density * v_ti * u / 4 * bracket
    np.testing.assert_allclose(actual, expected, rtol=3e-6)


def special_erf(value):
    import math
    return math.erf(value)


def test_polynomial_force_gradient_matches_finite_difference():
    coefficients = jnp.asarray(np.linspace(-0.02, 0.03, 15), jnp.float32)
    position = np.asarray([[0.3, 0.4, 0.2]], np.float32)
    value, gradient = polynomial_value_and_gradient(position, coefficients, 0.8)
    h = 2.0e-4
    numerical = []
    for axis in range(3):
        plus, minus = position.copy(), position.copy()
        plus[0, axis] += h
        minus[0, axis] -= h
        vp, _ = polynomial_value_and_gradient(plus, coefficients, 0.8)
        vm, _ = polynomial_value_and_gradient(minus, coefficients, 0.8)
        numerical.append(float((vp[0] - vm[0]) / (2 * h)))
    np.testing.assert_allclose(np.asarray(gradient[0]), numerical, rtol=2e-3, atol=2e-4)
    assert np.isfinite(float(value[0]))


def test_collision_event_replaces_or_preserves_complete_vector():
    velocities = jnp.arange(300, dtype=jnp.float32).reshape(100, 3)
    result, events = LegacyChargeExchangeCollision().apply(
        jax.random.PRNGKey(7), velocities, 2.0, 0.5
    )
    result = np.asarray(result)
    events = np.asarray(events)
    original = np.asarray(velocities)
    np.testing.assert_array_equal(result[~events], original[~events])
    assert np.all(np.any(result[events] != original[events], axis=1))


def test_cic_conserves_superion_weight_in_slice():
    positions = jnp.asarray(
        [[-0.9, 0.0, -0.9], [0.0, 0.1, 0.0], [1.0, 0.0, 1.0]],
        jnp.float32,
    )
    x = jnp.linspace(-1.0, 1.0, 5)
    z = jnp.linspace(-1.0, 1.0, 5)
    thickness = 0.1
    density = plane_cic_density_snapshot_m3(positions, x, z, thickness, 4.0)
    dx = float(x[1] - x[0])
    dz = float(z[1] - z[0])
    # The middle particle is outside |y| <= thickness/2.
    np.testing.assert_allclose(np.sum(np.asarray(density)) * dx * dz * thickness, 8.0)


def test_background_fit_is_deterministic_and_cacheable(tmp_path):
    config = PolynomialBackgroundConfig(
        seed=4, sample_count=256, source_block_size=64, cache_dir=tmp_path
    )
    first = build_polynomial_background(1.0, 2.0, 0.5, 32, config)
    second = build_polynomial_background(1.0, 2.0, 0.5, 32, config)
    np.testing.assert_array_equal(first.coefficients, second.coefficients)
    assert first.cache_key == second.cache_key
    assert np.isfinite(first.relative_rms_error)


def test_overlapping_dust_is_rejected():
    simulation = SimulationConfigSI(1.0, 2.0, 4, 1, 1e-4, force_backend="jax")
    dust = (
        DustParticleSI((0.0, 0.0, 0.0), 0.2),
        DustParticleSI((0.3, 0.0, 0.0), 0.2),
    )
    with pytest.raises(ValueError, match="overlap"):
        validate_system(plasma(), simulation, dust, None)


def test_collision_probability_above_one_is_rejected():
    driven = FieldDrivenPlasmaSI(300, 30_000, 1e10, 1.0, 20.0, 6.63e-26)
    simulation = SimulationConfigSI(1.0, 2.0, 4, 1, 0.1, force_backend="jax")
    with pytest.raises(ValueError, match="must not exceed 1"):
        validate_system(driven, simulation, (), None)


def test_first_completed_step_updates_charge_and_time_index():
    simulation = SimulationConfigSI(
        radius_m=1.0,
        height_m=2.0,
        particle_count=1,
        steps=1,
        dt_s=0.1,
        seed=3,
        force_backend="jax",
        step_chunk_size=1,
    )
    dust = DustParticleSI(
        (0.0, 0.0, 0.0), 0.2, initial_charge_C=-1.0e-18,
        charge_mode="self_consistent",
    )
    engine = OpenDustJAX(
        plasma(), simulation, [dust],
        background=PolynomialBackgroundConfig(enabled=False),
    )
    state = SimulationState(
        positions_m=jnp.asarray([[0.21, 0.0, 0.0]], jnp.float32),
        velocities_m_s=jnp.asarray([[-1.0, 0.0, 0.0]], jnp.float32),
        forces_N=jnp.zeros((1, 3), jnp.float32),
        dust_charges_C=jnp.asarray([-1.0e-18], jnp.float32),
        step=jnp.asarray(0, jnp.int32),
        prng_key=jax.random.PRNGKey(12),
    )
    result = engine.run(state)
    np.testing.assert_allclose(result.times_s, [0.0, 0.1])
    assert result.dust_charges_C.shape == (2, 1)
    assert result.dust_charges_C[1, 0] != result.dust_charges_C[0, 0]
    assert int(result.final_state.step) == 1


def test_checkpoint_keeps_positions_velocities_and_rng_separate(tmp_path):
    state = SimulationState(
        jnp.asarray([[1.0, 2.0, 3.0]], jnp.float32),
        jnp.asarray([[4.0, 5.0, 6.0]], jnp.float32),
        jnp.asarray([[7.0, 8.0, 9.0]], jnp.float32),
        jnp.asarray([-2.0], jnp.float32),
        jnp.asarray(17, jnp.int32),
        jax.random.PRNGKey(9),
    )
    path = tmp_path / "restart.npz"
    save_checkpoint(path, state, np.arange(15), {"case": "test"})
    restored, coefficients, metadata = load_checkpoint(path)
    np.testing.assert_array_equal(restored.positions_m, state.positions_m)
    np.testing.assert_array_equal(restored.velocities_m_s, state.velocities_m_s)
    np.testing.assert_array_equal(restored.prng_key, state.prng_key)
    np.testing.assert_array_equal(coefficients, np.arange(15))
    assert metadata == {"case": "test"}


def test_coincident_distinct_pairs_are_finite_in_reference_backend():
    forces = softened_yukawa_forces(
        jnp.zeros((2, 3), jnp.float32), 1.0, 1.0, 1.0e-3, block_size=2
    )
    np.testing.assert_array_equal(np.asarray(forces), np.zeros((2, 3), np.float32))


def test_split_run_matches_continuous_run(tmp_path):
    common = dict(
        radius_m=1.0,
        height_m=2.0,
        particle_count=4,
        dt_s=1.0e-5,
        seed=18,
        force_backend="jax",
        step_chunk_size=2,
    )
    background = PolynomialBackgroundConfig(enabled=False)
    full_engine = OpenDustJAX(
        plasma(), SimulationConfigSI(steps=4, **common), background=background
    )
    split_engine = OpenDustJAX(
        plasma(), SimulationConfigSI(steps=2, **common), background=background
    )
    initial = full_engine.initial_state()
    continuous = full_engine.run(initial).final_state
    first = split_engine.run(initial).final_state
    checkpoint = tmp_path / "split-restart"
    save_checkpoint(
        checkpoint, first, split_engine.background.coefficients, {"step": 2}
    )
    resumed = split_engine.run_from_checkpoint(checkpoint).final_state
    np.testing.assert_allclose(resumed.positions_m, continuous.positions_m, rtol=0, atol=0)
    np.testing.assert_allclose(resumed.velocities_m_s, continuous.velocities_m_s, rtol=0, atol=0)
    np.testing.assert_array_equal(resumed.prng_key, continuous.prng_key)
    assert int(resumed.step) == 4


def test_plane_diagnostics_are_averaged_and_written(tmp_path):
    simulation = SimulationConfigSI(
        radius_m=1.0,
        height_m=2.0,
        particle_count=4,
        steps=2,
        dt_s=1.0e-5,
        seed=5,
        force_backend="jax",
        step_chunk_size=2,
    )
    diagnostic = PlaneDiagnosticConfig(
        nx=4,
        nz=5,
        slice_thickness_m=0.2,
        start_step=1,
        interval=1,
        output_prefix=tmp_path / "plane",
        source_block_size=2,
    )
    warning_context = (
        pytest.warns(RuntimeWarning, match="x64")
        if not jax.config.x64_enabled
        else _no_warning()
    )
    with warning_context:
        engine = OpenDustJAX(
            plasma(),
            simulation,
            background=PolynomialBackgroundConfig(enabled=False),
            diagnostics=diagnostic,
        )
        result = engine.run()
    with np.load(result.diagnostic_files[0]) as fields:
        assert fields["potential_V"].shape == (5, 4)
        assert fields["ion_density_m3"].shape == (5, 4)
        assert int(fields["sample_count"]) == 2
        np.testing.assert_array_equal(fields["sample_steps"], [1, 2])


class _no_warning:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False
