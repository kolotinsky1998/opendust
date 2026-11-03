from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_ROOT = REPO_ROOT / "examples" / "OpenDust2023"
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from common import ELEMENTARY_CHARGE_C, averaging_start


def load_config(figure: str):
    path = EXAMPLE_ROOT / figure / "config.py"
    spec = importlib.util.spec_from_file_location(f"opendust2023_{figure}_config", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def arguments(case_index: int, output_dir: Path):
    return SimpleNamespace(
        case_index=case_index,
        particles=None,
        steps=None,
        background_samples=None,
        chunk_size=100,
        seed=2017,
        output_dir=output_dir,
    )


def test_case_arrays_and_charge_signs_match_legacy_examples():
    figure6 = load_config("Figure6")
    figure8 = load_config("Figure8")
    figure10 = load_config("Figure10")
    figure11 = load_config("Figure11")

    assert figure6.CHARGE_MODES == ("self_consistent", "oml")
    assert figure8.CASE_COUNT == 12
    assert figure8.MACH_VALUES[0] == pytest.approx(0.19679)
    assert figure8.MACH_VALUES[-1] == pytest.approx(2.0)
    assert figure8.INITIAL_CHARGE_E < 0.0
    assert figure10.CASE_COUNT == 13
    assert figure10.COLLISION_FREQUENCIES_S[-1] == pytest.approx(1.61113688e10)
    assert figure10.ELECTRIC_FIELDS_V_M[0] == pytest.approx(2.5852)
    assert figure11.CASE_COUNT == 7
    assert figure11.FIXED_CHARGE_E < 0.0


def test_production_configurations_and_averaging_windows(tmp_path):
    pytest.importorskip("jax")
    expected = {
        "Figure6": (2**16, 3000, 3.5148240854e-9, 1000),
        "Figure7": (2**16, 3000, 3.5148240854e-9, 1000),
        "Figure8": (2**16, 5000, 3.5148240854e-9, 3000),
        "Figure10": (2**17, 200_000, 1.0e-11, 150_000),
        "Figure11": (2**17, 50_000, 1.0e-9, 10_000),
    }
    for figure, values in expected.items():
        config = load_config(figure)
        setup = config.build_case(arguments(0, tmp_path), tmp_path)
        assert setup.simulation.particle_count == values[0]
        assert setup.simulation.steps == values[1]
        assert setup.simulation.dt_s == pytest.approx(values[2])
        assert setup.average_start_step == values[3]
        assert setup.simulation.force_backend == "cuda"
        assert setup.background.sample_count == 1_357_168
        assert all(dust.initial_charge_C <= 0.0 for dust in setup.dust_particles)


def test_smoke_override_keeps_nonempty_average_window():
    assert averaging_start(150_000, 10) == 5
    assert averaging_start(3000, 5000) == 3000


@pytest.mark.parametrize(
    ("figure", "array_spec"),
    [
        ("Figure6", "0-1"),
        ("Figure7", None),
        ("Figure8", "0-11"),
        ("Figure10", "0-12"),
        ("Figure11", "0-6"),
    ],
)
def test_sbatch_resources_and_arrays(figure, array_spec):
    text = (EXAMPLE_ROOT / figure / "submit.sbatch").read_text(encoding="utf-8")
    assert "#SBATCH --account=proj_1827" in text
    assert "#SBATCH --partition=rocky" in text
    assert "#SBATCH --time=1-00:00:00" in text
    assert "#SBATCH --gpus=1" in text
    if array_spec is None:
        assert "#SBATCH --array=" not in text
    else:
        assert f"#SBATCH --array={array_spec}" in text


def test_figure8_plot_from_synthetic_complete_results(tmp_path):
    pytest.importorskip("matplotlib")
    config = load_config("Figure8")
    for index, mach in enumerate(config.MACH_VALUES):
        case_dir = tmp_path / f"case_{index:02d}"
        case_dir.mkdir()
        summary = {
            "case": {"mach": mach},
            "mean_orbit_force_N": [[0.0, 0.0, (index + 1) * 1.0e-12]],
            "mean_collection_force_N": [[0.0, 0.0, 0.5e-12]],
        }
        (case_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        np.savez(case_dir / "history.npz", time_s=np.asarray([0.0]))
        (case_dir / ".done.json").write_text("{}", encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            str(EXAMPLE_ROOT / "Figure8" / "plot.py"),
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
    )
    assert (tmp_path / "aggregate.csv").is_file()
    assert (tmp_path / "figure8_ion_drag_vs_mach.png").is_file()
    assert (tmp_path / "figure8_ion_drag_vs_mach.pdf").is_file()
