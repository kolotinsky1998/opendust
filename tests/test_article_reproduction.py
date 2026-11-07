import json
import math
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from opendust_jax.article_reproduction import (
    ARTICLE_CASES,
    ARTICLE_DUST_RADIUS_M,
    ARTICLE_ELECTRON_TEMPERATURE_K,
    ARTICLE_ION_TEMPERATURE_K,
    ARTICLE_PRESSURE_PA,
    ARTICLE_REFERENCE_DENSITY_M3,
    ARTICLE_ION_MASS_KG,
    ArticleReproductionConfig,
    PaperElectricField,
    article_drift_mach,
    completed_artifact_matches,
    fit_required_field,
    fit_required_field_constrained,
    normalized_force_imbalance,
    paper_z_m,
    required_field_V_m,
    resolve_article_time_grid,
    save_iteration_artifact,
    simulation_z_m,
)
from opendust_jax.collisions import ArgonNullCollisionModel
from opendust_jax.models import (
    ELEMENTARY_CHARGE_C,
    EPSILON_0_F_M,
    ArgonPresheathPlasmaSI,
    CollisionSubcyclingConfig,
)
from opendust_jax.timestepping import resolve_collision_substeps
import opendust_jax.article_reproduction as article_reproduction


def test_paper_coordinate_and_field_round_trip():
    height = 8.0e-3
    z_sim = np.asarray([-3.0e-3, 0.0, 2.0e-3])
    np.testing.assert_allclose(simulation_z_m(paper_z_m(z_sim, height), height), z_sim)
    paper = PaperElectricField(0.646, -1.5, 0.150)
    simulation = paper.to_simulation(
        height,
        z_min_m=-3.2e-3,
        z_max_m=4.0e-3,
        enforce_presheath_constraints=False,
    )
    restored = PaperElectricField.from_simulation(simulation, height)
    np.testing.assert_allclose(
        [restored.E0_V_mm, restored.alpha_V_mm2, restored.beta_V_mm3],
        [paper.E0_V_mm, paper.alpha_V_mm2, paper.beta_V_mm3],
        rtol=2e-14,
        atol=2e-14,
    )
    for z in z_sim:
        expected = -paper.value_V_mm(float(paper_z_m(z, height) * 1e3)) * 1e3
        assert simulation.value_V_m(float(z)) == pytest.approx(expected)


def test_published_quadratic_field_is_strictly_admissible_across_dust_stack():
    lambda_de = 1.66227e-3
    height = 5.0 * lambda_de
    field = PaperElectricField(0.646, -1.5, 0.150).to_simulation(
        height,
        z_min_m=-1.94 * lambda_de,
        z_max_m=-0.33 * lambda_de,
    )
    field.validate_for_domain(height, lowest_dust_z_m=-1.94 * lambda_de)


def test_unweighted_linear_and_quadratic_fits_recover_coefficients():
    height = 8.0e-3
    z = np.linspace(-3.0e-3, 0.0, 7)
    for degree, expected in (
        (1, PaperElectricField(-4.2, 0.23, 0.0)),
        (2, PaperElectricField(0.65, -1.5, 0.15)),
    ):
        paper_z_mm = paper_z_m(z, height) * 1e3
        sim_field = -np.asarray(expected.value_V_mm(paper_z_mm)) * 1e3
        actual = fit_required_field(z, sim_field, height, degree)
        np.testing.assert_allclose(
            [actual.E0_V_mm, actual.alpha_V_mm2, actual.beta_V_mm3],
            [expected.E0_V_mm, expected.alpha_V_mm2, expected.beta_V_mm3],
            rtol=1e-12,
            atol=1e-12,
        )


def test_article_time_grid_has_exact_outer_steps_duration_and_burn_in(tmp_path):
    config = ArticleReproductionConfig(output_dir=tmp_path)
    omega_pi = math.sqrt(
        ARTICLE_REFERENCE_DENSITY_M3 * ELEMENTARY_CHARGE_C**2
        / (EPSILON_0_F_M * ARTICLE_ION_MASS_KG)
    )
    grid = resolve_article_time_grid(config, omega_pi)
    assert grid.outer_steps == 250_000
    assert grid.burn_in_step == 50_000
    assert grid.duration_s == pytest.approx(83.0 * 2.0 * math.pi / omega_pi)
    assert grid.outer_steps * grid.dt_s == pytest.approx(grid.duration_s)
    assert grid.dt_s == pytest.approx(9.97824065802749e-10)


def test_article_grid_resolves_collision_and_charging_limits(tmp_path):
    config = ArticleReproductionConfig(output_dir=tmp_path)
    omega_pi = math.sqrt(
        ARTICLE_REFERENCE_DENSITY_M3 * ELEMENTARY_CHARGE_C**2
        / (EPSILON_0_F_M * ARTICLE_ION_MASS_KG)
    )
    grid = resolve_article_time_grid(config, omega_pi)
    height = 5.0 * math.sqrt(
        EPSILON_0_F_M
        * 1.380649e-23
        * ARTICLE_ELECTRON_TEMPERATURE_K
        / (ELEMENTARY_CHARGE_C**2 * ARTICLE_REFERENCE_DENSITY_M3)
    )
    for case in ARTICLE_CASES:
        plasma = ArgonPresheathPlasmaSI(
            ARTICLE_ION_TEMPERATURE_K,
            ARTICLE_ELECTRON_TEMPERATURE_K,
            ARTICLE_REFERENCE_DENSITY_M3,
            ARTICLE_PRESSURE_PA,
            case.initial_mach,
            PaperElectricField(-case.constant_field_V_m / 1.0e3).to_simulation(
                height, enforce_presheath_constraints=False
            ),
            ARTICLE_ION_MASS_KG,
        )
        collision = ArgonNullCollisionModel(plasma)
        substeps, collision_dt, _, _ = resolve_collision_substeps(
            plasma,
            grid.dt_s,
            CollisionSubcyclingConfig(),
            collision,
        )
        candidate_probability = 1.0 - math.exp(
            -collision.maximum_frequency_s * collision_dt
        )
        assert substeps == 4
        assert candidate_probability <= 0.1 + 1.0e-12
        velocity_limit = abs(plasma.flow_speed_m_s) + 4.0 * plasma.ion_thermal_speed_m_s
        charging_substeps = math.ceil(
            grid.dt_s * velocity_limit / (0.1 * ARTICLE_DUST_RADIUS_M)
        )
        assert (
            grid.dt_s * velocity_limit / charging_substeps
            <= 0.1 * ARTICLE_DUST_RADIUS_M
        )


def test_constrained_linear_fit_clamps_negative_slope():
    height = 8.0e-3
    z = np.linspace(-3.0e-3, -1.0e-3, 7)
    raw = PaperElectricField(-1.0, -0.4, 0.0)
    required = -np.asarray(raw.value_V_mm(paper_z_m(z, height) * 1.0e3)) * 1.0e3
    fit = fit_required_field_constrained(
        z,
        required,
        height,
        1,
        active_z_min_m=float(z[0]),
        active_z_max_m=float(z[-1]),
    )
    assert fit.unconstrained_field.alpha_V_mm2 < 0.0
    assert fit.field.alpha_V_mm2 == 0.0
    assert fit.constraints_active
    fit.field.to_simulation(
        height, z_min_m=float(z[0]), z_max_m=float(z[-1])
    ).validate_for_domain(height, lowest_dust_z_m=float(z[0]))


def test_constrained_quadratic_fit_projects_into_strict_region():
    height = 8.0e-3
    z = np.linspace(-3.0e-3, -1.0e-3, 7)
    invalid = PaperElectricField(-2.0, 0.4, -0.08)
    required = -np.asarray(invalid.value_V_mm(paper_z_m(z, height) * 1.0e3)) * 1.0e3
    fit = fit_required_field_constrained(
        z,
        required,
        height,
        2,
        active_z_min_m=float(z[0]),
        active_z_max_m=float(z[-1]),
    )
    assert fit.field.beta_V_mm3 > 0.0
    assert fit.constraints_active
    simulation = fit.field.to_simulation(
        height, z_min_m=float(z[0]), z_max_m=float(z[-1])
    )
    simulation.validate_for_domain(height, lowest_dust_z_m=float(z[0]))


def test_constrained_quadratic_fit_uses_feasible_fallback(monkeypatch):
    height = 8.0e-3
    z = np.linspace(-3.0e-3, -1.0e-3, 7)
    invalid = PaperElectricField(-2.0, 0.4, -0.08)
    required = -np.asarray(invalid.value_V_mm(paper_z_m(z, height) * 1.0e3)) * 1.0e3

    def failed_optimizer(*args, **kwargs):
        return SimpleNamespace(
            success=False,
            x=np.asarray([np.nan, np.nan, np.nan]),
            fun=np.nan,
            message="synthetic failure",
        )

    monkeypatch.setattr(article_reproduction.optimize, "minimize", failed_optimizer)
    fit = fit_required_field_constrained(
        z,
        required,
        height,
        2,
        active_z_min_m=float(z[0]),
        active_z_max_m=float(z[-1]),
    )
    assert fit.fallback_used
    assert not fit.optimizer_success
    fit.field.to_simulation(
        height, z_min_m=float(z[0]), z_max_m=float(z[-1])
    ).validate_for_domain(height, lowest_dust_z_m=float(z[0]))


def test_force_balance_and_normalized_imbalance_are_consistent():
    charge = np.asarray([-2.0e-15, -2.5e-15])
    orbit = np.asarray([1.0e-12, -2.0e-13])
    collection = np.asarray([2.0e-13, 3.0e-13])
    dust_dust = np.asarray([-1.0e-13, 4.0e-13])
    field = required_field_V_m(charge, orbit, collection, dust_dust)
    imbalance = normalized_force_imbalance(
        charge, field, orbit, collection, dust_dust
    )
    np.testing.assert_allclose(imbalance, 0.0, atol=2e-16)


def test_equation_15_is_monotonic_and_uses_field_magnitude():
    values = [article_drift_mach(field) for field in (4_790.0, 16_000.0, 32_500.0)]
    assert 0.0 < values[0] < values[1] < values[2]
    np.testing.assert_allclose(values, [0.46, 0.92, 1.40], rtol=0.10)
    assert article_drift_mach(-16_000.0) == pytest.approx(values[1])


def test_atomic_artifact_compatibility_check(tmp_path):
    prefix = tmp_path / "stage"
    npz, metadata = save_iteration_artifact(
        prefix, {"value": [1.0, 2.0]}, {"status": "complete", "fingerprint": "abc"}
    )
    assert npz.is_file() and metadata.is_file()
    assert completed_artifact_matches(prefix, "abc")
    assert not completed_artifact_matches(prefix, "different")
    with np.load(npz, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["value"], [1.0, 2.0])
