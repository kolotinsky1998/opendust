import importlib.util
import math
from pathlib import Path
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = REPO_ROOT / "examples" / "ChainRotationExperiment"
WORKFLOW_PATH = EXAMPLE_DIR / "workflow.py"
SPEC = importlib.util.spec_from_file_location("chain_rotation_workflow", WORKFLOW_PATH)
assert SPEC is not None and SPEC.loader is not None
workflow = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = workflow
SPEC.loader.exec_module(workflow)


def config():
    return workflow.load_config(EXAMPLE_DIR / "chain_config.json")


def test_dust_mass_and_grid_definition():
    value = config()
    assert value.dust_mass_kg == pytest.approx(9.031352385769511e-13)
    selected = workflow.cases(value)
    assert len(selected) == 9
    assert selected[0].name == "Te_3eV_n_1e14"
    assert selected[-1].name == "Te_5eV_n_1e16"


def test_coordinates_are_sorted_centered_and_preserved():
    value = config()
    case = workflow.cases(value)[0]
    geometry = workflow.build_geometry(value, case)
    assert geometry.particle_ids.tolist() == list(range(1, 9))
    assert np.all(np.diff(geometry.simulation_positions_m[:, 2]) > 0.0)
    np.testing.assert_allclose(
        np.mean(geometry.simulation_positions_m[:, :2], axis=0), 0.0, atol=1e-20
    )
    np.testing.assert_allclose(
        np.diff(geometry.simulation_positions_m, axis=0),
        np.diff(geometry.laboratory_positions_m, axis=0),
    )
    assert geometry.chain_span_m == pytest.approx(0.002367612750035008)
    assert geometry.maximum_radial_offset_m == pytest.approx(
        0.0001509633048611962
    )
    assert geometry.radius_m == pytest.approx(0.0020823580901623653)
    assert geometry.height_m == pytest.approx(0.010300772188272902)


def test_constant_field_roundtrips_to_mach_half():
    value = config()
    for case in workflow.cases(value)[::4]:
        field = workflow.constant_field_for_mach(value, case)
        mach = workflow.article_drift_mach(
            field,
            pressure_Pa=value.pressure_Pa,
            neutral_temperature_K=value.neutral_temperature_K,
            electron_temperature_K=case.electron_temperature_K,
            ion_mass_kg=value.ion_mass_kg,
        )
        assert mach == pytest.approx(0.5, rel=1e-12)


def test_particle_count_policy_and_superion_resolution():
    value = config()
    selected = workflow.cases(value)
    for case in selected:
        expected = 2**17 if case.reference_density_m3 == 1.0e14 else 2**16
        assert workflow.particle_count(value, case) == expected
        geometry = workflow.build_geometry(value, case)
        real_ions = (
            case.reference_density_m3
            * math.pi
            * geometry.radius_m**2
            * geometry.height_m
        )
        weight = real_ions / expected
        initial_charge = (
            value.dust_mass_kg
            * value.gravity_m_s2
            / workflow.constant_field_for_mach(value, case)
        )
        ratio = workflow.ELEMENTARY_CHARGE_C * weight / initial_charge
        assert ratio < 0.0243


def test_all_slurm_jobs_use_rocky_partition():
    for filename in ("slurm_array.sh", "slurm_postprocess.sh"):
        text = (EXAMPLE_DIR / filename).read_text(encoding="utf-8")
        assert "#SBATCH --partition=rocky" in text
    array = (EXAMPLE_DIR / "slurm_array.sh").read_text(encoding="utf-8")
    assert "#SBATCH --array=0-8" in array
    assert "#SBATCH --time=04:00:00" in array
    assert "#SBATCH --gres=gpu:1" in array
