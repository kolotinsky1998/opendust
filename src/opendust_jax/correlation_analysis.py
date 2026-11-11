"""Correlation-aware statistics for OpenDust production histories."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class CorrelatedMeanEstimate:
    """Mean and uncertainty estimated from one stationary time series."""

    sample_count: int
    mean: float
    standard_deviation: float
    standard_error: float
    confidence_low: float
    confidence_high: float
    confidence_level: float
    integrated_autocorrelation_steps: float
    integrated_autocorrelation_time_s: float
    statistical_inefficiency: float
    effective_sample_size: float
    e_folding_lag_steps: int | None
    e_folding_time_s: float | None
    first_zero_lag_steps: int | None
    first_zero_time_s: float | None
    geyer_cutoff_lag_steps: int
    stationarity_z_score: float | None
    stationarity_warning: bool
    low_effective_sample_size_warning: bool
    constant_series_warning: bool

    def to_dict(self) -> dict[str, float | int | bool | None]:
        return asdict(self)


def normalized_autocorrelation(values: np.ndarray) -> np.ndarray:
    """Return the unbiased normalized autocorrelation using an FFT."""

    series = _validated_series(values)
    centered = series - np.mean(series)
    variance_sum = float(np.dot(centered, centered))
    if variance_sum == 0.0:
        return np.ones(series.size, dtype=float)

    fft_size = 1 << (2 * series.size - 1).bit_length()
    transformed = np.fft.rfft(centered, n=fft_size)
    covariance = np.fft.irfft(
        transformed * np.conjugate(transformed), n=fft_size
    )[: series.size]
    covariance /= np.arange(series.size, 0, -1, dtype=float)
    return covariance / covariance[0]


def geyer_initial_positive_time(
    autocorrelation: np.ndarray,
) -> tuple[float, int]:
    """Estimate integrated autocorrelation steps with Geyer's IPS rule.

    The returned value uses the convention ``tau = 1/2 + sum(rho[k])``.
    Therefore the statistical inefficiency is ``g = 2 * tau``.
    """

    rho = np.asarray(autocorrelation, dtype=float)
    if rho.ndim != 1 or rho.size < 2 or not np.all(np.isfinite(rho)):
        raise ValueError("autocorrelation must be a finite one-dimensional array")
    if not np.isclose(rho[0], 1.0):
        raise ValueError("autocorrelation must be normalized to one at zero lag")

    paired_sums = rho[1:-1:2] + rho[2::2]
    nonpositive = np.flatnonzero(paired_sums <= 0.0)
    positive_pair_count = (
        int(nonpositive[0]) if nonpositive.size else int(paired_sums.size)
    )
    cutoff_lag = 2 * positive_pair_count
    tau_steps = 0.5 + float(np.sum(rho[1 : cutoff_lag + 1]))
    return max(0.5, tau_steps), cutoff_lag


def estimate_correlated_mean(
    values: np.ndarray,
    dt_s: float,
    *,
    confidence_level: float = 0.95,
    minimum_effective_samples: float = 20.0,
    check_stationarity: bool = True,
) -> CorrelatedMeanEstimate:
    """Estimate a mean and CI after accounting for serial correlation."""

    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("dt_s must be finite and positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")

    series = _validated_series(values)
    sample_count = int(series.size)
    mean = float(np.mean(series))
    standard_deviation = float(np.std(series, ddof=1))
    constant = standard_deviation == 0.0

    if constant:
        rho = np.ones(sample_count, dtype=float)
        tau_steps = 0.5
        cutoff_lag = 0
        e_folding_lag = None
        first_zero_lag = None
    else:
        rho = normalized_autocorrelation(series)
        tau_steps, cutoff_lag = geyer_initial_positive_time(rho)
        e_folding_lag = _first_lag_at_or_below(rho, math.exp(-1.0))
        first_zero_lag = _first_lag_at_or_below(rho, 0.0)

    statistical_inefficiency = max(1.0, 2.0 * tau_steps)
    effective_sample_size = min(
        float(sample_count), sample_count / statistical_inefficiency
    )
    standard_error = standard_deviation * math.sqrt(
        statistical_inefficiency / sample_count
    )
    degrees_of_freedom = max(1.0, effective_sample_size - 1.0)
    probability = 0.5 * (1.0 + confidence_level)
    multiplier = float(stats.t.ppf(probability, degrees_of_freedom))
    confidence_half_width = multiplier * standard_error

    stationarity_z_score = None
    stationarity_warning = False
    if check_stationarity and not constant and sample_count >= 8:
        midpoint = sample_count // 2
        first = estimate_correlated_mean(
            series[:midpoint],
            dt_s,
            confidence_level=confidence_level,
            minimum_effective_samples=minimum_effective_samples,
            check_stationarity=False,
        )
        second = estimate_correlated_mean(
            series[midpoint:],
            dt_s,
            confidence_level=confidence_level,
            minimum_effective_samples=minimum_effective_samples,
            check_stationarity=False,
        )
        denominator = math.hypot(first.standard_error, second.standard_error)
        if denominator > 0.0:
            stationarity_z_score = abs(first.mean - second.mean) / denominator
            stationarity_warning = stationarity_z_score > 1.96

    return CorrelatedMeanEstimate(
        sample_count=sample_count,
        mean=mean,
        standard_deviation=standard_deviation,
        standard_error=standard_error,
        confidence_low=mean - confidence_half_width,
        confidence_high=mean + confidence_half_width,
        confidence_level=confidence_level,
        integrated_autocorrelation_steps=tau_steps,
        integrated_autocorrelation_time_s=tau_steps * dt_s,
        statistical_inefficiency=statistical_inefficiency,
        effective_sample_size=effective_sample_size,
        e_folding_lag_steps=e_folding_lag,
        e_folding_time_s=(None if e_folding_lag is None else e_folding_lag * dt_s),
        first_zero_lag_steps=first_zero_lag,
        first_zero_time_s=(None if first_zero_lag is None else first_zero_lag * dt_s),
        geyer_cutoff_lag_steps=cutoff_lag,
        stationarity_z_score=stationarity_z_score,
        stationarity_warning=stationarity_warning,
        low_effective_sample_size_warning=(
            effective_sample_size < minimum_effective_samples
        ),
        constant_series_warning=constant,
    )


def build_observable_histories(
    arrays: Mapping[str, np.ndarray], burn_in_step: int
) -> dict[str, np.ndarray]:
    """Extract post-burn-in charge and axial force histories from an artifact."""

    vector = build_vector_observable_histories(arrays, burn_in_step)
    return {
        "charge_C": vector["charge_C"],
        "orbit_z_N": vector["orbit_z_N"],
        "collection_z_N": vector["collection_z_N"],
        "dust_dust_z_N": vector["dust_dust_z_N"],
        "total_z_N": vector["total_nonexternal_z_N"],
    }


def build_vector_observable_histories(
    arrays: Mapping[str, np.ndarray], burn_in_step: int
) -> dict[str, np.ndarray]:
    """Extract post-burn-in charge and all Cartesian force histories."""

    required = (
        "charge_history_C",
        "orbit_force_history_N",
        "collection_force_history_N",
        "dust_dust_force_history_N",
    )
    missing = [name for name in required if name not in arrays]
    if missing:
        raise KeyError(f"missing production histories: {', '.join(missing)}")

    charge = np.asarray(arrays["charge_history_C"], dtype=float)
    if charge.ndim != 2:
        raise ValueError("charge_history_C must have shape (steps, dust particles)")
    if burn_in_step < 0 or burn_in_step >= charge.shape[0] - 1:
        raise ValueError("burn_in_step leaves fewer than two samples")

    components = ("x", "y", "z")
    forces: dict[str, np.ndarray] = {}
    vector_forces: dict[str, np.ndarray] = {}
    for force_name, input_name in (
        ("orbit", "orbit_force_history_N"),
        ("collection", "collection_force_history_N"),
        ("dust_dust", "dust_dust_force_history_N"),
    ):
        values = np.asarray(arrays[input_name], dtype=float)
        if values.shape != (charge.shape[0], charge.shape[1], 3):
            raise ValueError(f"{input_name} has an incompatible shape")
        vector_forces[force_name] = values[burn_in_step:]
        for axis, component in enumerate(components):
            forces[f"{force_name}_{component}_N"] = values[burn_in_step:, :, axis]

    histories = {"charge_C": charge[burn_in_step:], **forces}
    total = sum(vector_forces.values())
    for axis, component in enumerate(components):
        histories[f"total_nonexternal_{component}_N"] = total[:, :, axis]
    return histories


def _validated_series(values: np.ndarray) -> np.ndarray:
    series = np.asarray(values, dtype=float)
    if series.ndim != 1:
        raise ValueError("time series must be one-dimensional")
    if series.size < 4:
        raise ValueError("time series must contain at least four samples")
    if not np.all(np.isfinite(series)):
        raise ValueError("time series contains non-finite values")
    return series


def _first_lag_at_or_below(
    autocorrelation: np.ndarray, threshold: float
) -> int | None:
    matches = np.flatnonzero(autocorrelation[1:] <= threshold)
    return None if matches.size == 0 else int(matches[0] + 1)
