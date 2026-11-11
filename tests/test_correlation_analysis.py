import math
from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from opendust_jax.correlation_analysis import (
    build_observable_histories,
    build_vector_observable_histories,
    estimate_correlated_mean,
)


def test_white_noise_has_unit_statistical_inefficiency():
    rng = np.random.default_rng(20261110)
    estimate = estimate_correlated_mean(rng.normal(size=100_000), 0.25)
    assert 1.0 <= estimate.statistical_inefficiency < 1.2
    assert estimate.effective_sample_size > 80_000
    assert estimate.standard_error == pytest.approx(
        estimate.standard_deviation
        * math.sqrt(estimate.statistical_inefficiency / estimate.sample_count)
    )


def test_ar1_recovers_known_statistical_inefficiency():
    rng = np.random.default_rng(41)
    phi = 0.8
    noise = rng.normal(size=200_000)
    values = np.empty_like(noise)
    values[0] = noise[0] / math.sqrt(1.0 - phi**2)
    for index in range(1, values.size):
        values[index] = phi * values[index - 1] + noise[index]

    estimate = estimate_correlated_mean(values, 2.0e-3)
    expected = (1.0 + phi) / (1.0 - phi)
    assert estimate.statistical_inefficiency == pytest.approx(expected, rel=0.15)
    assert estimate.integrated_autocorrelation_time_s == pytest.approx(
        0.5 * expected * 2.0e-3, rel=0.15
    )


def test_constant_series_has_zero_uncertainty_and_explicit_warning():
    estimate = estimate_correlated_mean(np.full(20, -3.0), 1.0)
    assert estimate.mean == -3.0
    assert estimate.standard_error == 0.0
    assert estimate.confidence_low == -3.0
    assert estimate.confidence_high == -3.0
    assert estimate.statistical_inefficiency == 1.0
    assert estimate.constant_series_warning


@pytest.mark.parametrize(
    "values, message",
    [
        (np.ones(3), "at least four"),
        (np.asarray([1.0, 2.0, np.nan, 4.0]), "non-finite"),
    ],
)
def test_invalid_series_are_rejected(values, message):
    with pytest.raises(ValueError, match=message):
        estimate_correlated_mean(values, 1.0)


def test_total_force_is_built_before_statistical_analysis():
    charge = np.arange(10.0).reshape(5, 2)
    arrays = {"charge_history_C": charge}
    expected = np.zeros((5, 2))
    for index, name in enumerate(
        (
            "orbit_force_history_N",
            "collection_force_history_N",
            "dust_dust_force_history_N",
        ),
        start=1,
    ):
        force = np.zeros((5, 2, 3))
        force[:, :, 2] = index * charge
        arrays[name] = force
        expected += index * charge

    histories = build_observable_histories(arrays, burn_in_step=1)
    np.testing.assert_array_equal(histories["total_z_N"], expected[1:])
    np.testing.assert_array_equal(histories["charge_C"], charge[1:])


def test_vector_histories_include_every_force_component():
    charge = np.arange(10.0).reshape(5, 2)
    arrays = {"charge_history_C": charge}
    expected = np.zeros((5, 2, 3))
    for index, name in enumerate(
        (
            "orbit_force_history_N",
            "collection_force_history_N",
            "dust_dust_force_history_N",
        ),
        start=1,
    ):
        values = np.repeat(charge[:, :, None], 3, axis=2) * index
        arrays[name] = values
        expected += values

    histories = build_vector_observable_histories(arrays, burn_in_step=1)
    for axis, component in enumerate("xyz"):
        np.testing.assert_array_equal(
            histories[f"total_nonexternal_{component}_N"], expected[1:, :, axis]
        )
