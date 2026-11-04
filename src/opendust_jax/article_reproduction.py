"""Utilities for reproducing Phys. Rev. E 102, 043210 (2020).

The paper uses an axial coordinate opposite to the OpenDustJAX presheath
coordinate.  This module keeps that conversion explicit and centralizes the
force-balance and iteration bookkeeping used by the CUDA/Slurm workflow.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np

from opendust_jax.models import (
    BOLTZMANN_J_K,
    ELEMENTARY_CHARGE_C,
    EPSILON_0_F_M,
    QuadraticElectricFieldSI,
    SimulationResult,
)


ARTICLE_ELECTRON_TEMPERATURE_K = 58_022.0
ARTICLE_ION_TEMPERATURE_K = 290.0
ARTICLE_REFERENCE_DENSITY_M3 = 1.0e14
ARTICLE_PRESSURE_PA = 19.9984
ARTICLE_ION_MASS_KG = 6.6335209e-26
ARTICLE_DUST_RADIUS_M = 4.445e-6
ARTICLE_DUST_MASS_KG = 5.56e-13
ARTICLE_DUST_Z_OVER_LAMBDA = (-1.94, -1.65, -1.37, -1.13, -0.89, -0.62, -0.33)


@dataclass(frozen=True)
class ArticleCase:
    name: str
    initial_mach: float
    constant_field_V_m: float
    seed: int


ARTICLE_CASES = (
    ArticleCase("case_I", 0.46, 4_790.0, 10_041),
    ArticleCase("case_II", 0.92, 16_000.0, 20_041),
    ArticleCase("case_III", 1.40, 32_500.0, 30_041),
)


@dataclass(frozen=True)
class PaperElectricField:
    """Polynomial coefficients in the paper's V/mm convention."""

    E0_V_mm: float
    alpha_V_mm2: float = 0.0
    beta_V_mm3: float = 0.0

    def value_V_mm(self, z_paper_mm: float | np.ndarray) -> float | np.ndarray:
        z = np.asarray(z_paper_mm)
        value = self.E0_V_mm + self.alpha_V_mm2 * z + self.beta_V_mm3 * z**2
        return float(value) if value.ndim == 0 else value

    def to_simulation(
        self,
        height_m: float,
        *,
        z_min_m: float | None = None,
        z_max_m: float | None = None,
        enforce_presheath_constraints: bool = True,
    ) -> QuadraticElectricFieldSI:
        """Convert ``E_paper(z_paper)`` to the opposite OpenDust coordinate."""
        e0 = 1.0e3 * self.E0_V_mm
        alpha = 1.0e6 * self.alpha_V_mm2
        beta = 1.0e9 * self.beta_V_mm3
        half_height = 0.5 * height_m
        return QuadraticElectricFieldSI(
            E0_V_m=-(e0 + alpha * half_height + beta * half_height**2),
            alpha_V_m2=alpha + 2.0 * beta * half_height,
            beta_V_m3=-beta,
            z_min_m=z_min_m,
            z_max_m=z_max_m,
            enforce_presheath_constraints=enforce_presheath_constraints,
        )

    @classmethod
    def from_simulation(
        cls, field: QuadraticElectricFieldSI, height_m: float
    ) -> "PaperElectricField":
        half_height = 0.5 * height_m
        beta = -field.beta_V_m3
        alpha = field.alpha_V_m2 - 2.0 * beta * half_height
        e0 = -field.E0_V_m - alpha * half_height - beta * half_height**2
        return cls(e0 / 1.0e3, alpha / 1.0e6, beta / 1.0e9)


@dataclass(frozen=True)
class ArticleReproductionConfig:
    output_dir: Path | str
    particle_count: int = 2**16
    iterations: int = 5
    duration_plasma_periods: float = 83.0
    burn_in_plasma_periods: float = 16.6
    radius_screening_lengths: float = 1.25
    height_screening_lengths: float = 5.0
    force_backend: Literal["cuda", "jax"] = "cuda"
    diagnostic_nx: int = 81
    diagnostic_nz: int = 161
    diagnostic_interval: int = 100
    step_chunk_size: int = 100
    background_grid_spacing_screening_lengths: float = 0.1
    background_density_tolerance: float = 0.02
    background_flux_tolerance: float = 0.02
    background_consecutive_blocks: int = 3
    background_minimum_transit_times: float = 4.0
    background_maximum_transit_times: float = 100.0
    background_averaging_blocks: int = 10
    smoke_project_invalid_fields: bool = False

    def __post_init__(self) -> None:
        if self.particle_count <= 0 or self.iterations < 2:
            raise ValueError("particle_count must be positive and iterations at least two")
        if not 0.0 < self.burn_in_plasma_periods < self.duration_plasma_periods:
            raise ValueError("burn-in must be positive and shorter than the run")
        if self.force_backend not in ("cuda", "jax"):
            raise ValueError("force_backend must be 'cuda' or 'jax'")


def paper_z_m(z_sim_m: float | np.ndarray, height_m: float) -> float | np.ndarray:
    value = 0.5 * height_m - np.asarray(z_sim_m)
    return float(value) if value.ndim == 0 else value


def simulation_z_m(z_paper_m: float | np.ndarray, height_m: float) -> float | np.ndarray:
    value = 0.5 * height_m - np.asarray(z_paper_m)
    return float(value) if value.ndim == 0 else value


def article_sound_speed_m_s(
    electron_temperature_K: float = ARTICLE_ELECTRON_TEMPERATURE_K,
    ion_mass_kg: float = ARTICLE_ION_MASS_KG,
) -> float:
    return math.sqrt(BOLTZMANN_J_K * electron_temperature_K / ion_mass_kg)


def article_drift_mach(
    electric_field_V_m: float,
    *,
    pressure_Pa: float = ARTICLE_PRESSURE_PA,
    neutral_temperature_K: float = ARTICLE_ION_TEMPERATURE_K,
    electron_temperature_K: float = ARTICLE_ELECTRON_TEMPERATURE_K,
    ion_mass_kg: float = ARTICLE_ION_MASS_KG,
    argon_polarizability_m3: float = 1.6411e-30,
    A_eV_minus_quarter: float = 2.6,
    paper_impact_parameter_scale: float = 0.77,
) -> float:
    """Evaluate Eq. (15), including the paper's polarization cross section.

    ``argon_polarizability_m3`` is the quoted polarizability volume.  It is
    converted to SI polarizability before evaluating ``a`` in Eq. (13).
    The final scale preserves the mixed eV/SI convention used for the paper's
    quoted ``A`` and reproduces its three tabulated constant-field E/N points
    to better than ten percent.
    """
    neutral_density = pressure_Pa / (BOLTZMANN_J_K * neutral_temperature_K)
    polarizability_si = 4.0 * math.pi * EPSILON_0_F_M * argon_polarizability_m3
    a_si = polarizability_si * ELEMENTARY_CHARGE_C**2 / (
        2.0 * (4.0 * math.pi * EPSILON_0_F_M) ** 2
    )
    A_si = A_eV_minus_quarter * ELEMENTARY_CHARGE_C ** (-0.25)
    b_ex_m = paper_impact_parameter_scale * A_si * (4.0 * a_si) ** 0.25
    mobility_factor = (
        8.0
        / (3.0 * ELEMENTARY_CHARGE_C)
        * math.sqrt(ion_mass_kg / math.pi)
        * math.pi
        * b_ex_m**2
    )
    thermal_term = 3.0 * BOLTZMANN_J_K * neutral_temperature_K / ion_mass_kg
    reduced_field = abs(electric_field_V_m) / neutral_density
    velocity_squared = math.sqrt(
        thermal_term**2
        + 3.0 * reduced_field**2 / (mobility_factor**2 * ion_mass_kg)
    ) - thermal_term
    return math.sqrt(max(velocity_squared, 0.0)) / article_sound_speed_m_s(
        electron_temperature_K, ion_mass_kg
    )


def fit_required_field(
    z_sim_m: np.ndarray,
    required_field_sim_V_m: np.ndarray,
    height_m: float,
    degree: Literal[1, 2],
) -> PaperElectricField:
    """Unweighted fit in paper coordinates and paper coefficient units."""
    z_paper_mm = np.asarray(paper_z_m(z_sim_m, height_m)) * 1.0e3
    field_paper_V_mm = -np.asarray(required_field_sim_V_m) / 1.0e3
    coefficients = np.polynomial.polynomial.polyfit(
        z_paper_mm, field_paper_V_mm, degree
    )
    if degree == 1:
        coefficients = np.append(coefficients, 0.0)
    return PaperElectricField(*map(float, coefficients))


def required_field_V_m(
    mean_charge_C: np.ndarray,
    orbit_force_z_N: np.ndarray,
    collection_force_z_N: np.ndarray,
    dust_dust_force_z_N: np.ndarray,
    *,
    dust_mass_kg: float = ARTICLE_DUST_MASS_KG,
    gravity_m_s2: float = 9.81,
) -> np.ndarray:
    charge = np.asarray(mean_charge_C, dtype=np.float64)
    if np.any(charge == 0.0):
        raise ValueError("force balance is undefined for zero dust charge")
    other = (
        np.asarray(orbit_force_z_N)
        + np.asarray(collection_force_z_N)
        + np.asarray(dust_dust_force_z_N)
    )
    return -(dust_mass_kg * gravity_m_s2 + other) / charge


def normalized_force_imbalance(
    charge_C: np.ndarray,
    field_sim_V_m: np.ndarray,
    orbit_force_z_N: np.ndarray,
    collection_force_z_N: np.ndarray,
    dust_dust_force_z_N: np.ndarray,
    *,
    dust_mass_kg: float = ARTICLE_DUST_MASS_KG,
    gravity_m_s2: float = 9.81,
) -> np.ndarray:
    weight = dust_mass_kg * gravity_m_s2
    return (
        weight
        + np.asarray(orbit_force_z_N)
        + np.asarray(collection_force_z_N)
        + np.asarray(dust_dust_force_z_N)
        + np.asarray(charge_C) * np.asarray(field_sim_V_m)
    ) / weight


def summarize_production_result(
    result: SimulationResult,
    burn_in_step: int,
    field_sim_V_m: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute the iteration means/error bars and force-balance target."""
    if burn_in_step < 0 or burn_in_step >= len(result.times_s):
        raise ValueError("burn_in_step lies outside the production history")
    arrays = {
        "charge_C": np.asarray(result.dust_charges_C)[burn_in_step:],
        "orbit_z_N": np.asarray(result.dust_orbit_forces_N)[burn_in_step:, :, 2],
        "collection_z_N": np.asarray(result.dust_collection_forces_N)[burn_in_step:, :, 2],
        "dust_dust_z_N": np.asarray(result.dust_dust_forces_N)[burn_in_step:, :, 2],
    }
    summary: dict[str, np.ndarray] = {}
    for name, values in arrays.items():
        summary[f"mean_{name}"] = np.mean(values, axis=0)
        summary[f"std_{name}"] = np.std(values, axis=0, ddof=0)
    summary["required_field_sim_V_m"] = required_field_V_m(
        summary["mean_charge_C"],
        summary["mean_orbit_z_N"],
        summary["mean_collection_z_N"],
        summary["mean_dust_dust_z_N"],
    )
    summary["normalized_force_imbalance"] = normalized_force_imbalance(
        summary["mean_charge_C"],
        field_sim_V_m,
        summary["mean_orbit_z_N"],
        summary["mean_collection_z_N"],
        summary["mean_dust_dust_z_N"],
    )
    return summary


def artifact_fingerprint(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]


def save_iteration_artifact(
    prefix: Path | str,
    arrays: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> tuple[Path, Path]:
    """Atomically save a resumable NPZ+JSON stage pair."""
    base = Path(prefix).expanduser()
    base.parent.mkdir(parents=True, exist_ok=True)
    npz_path = base.with_suffix(".npz")
    json_path = base.with_suffix(".json")
    temporary_npz = npz_path.with_name(npz_path.name + ".tmp.npz")
    temporary_json = json_path.with_name(json_path.name + ".tmp")
    np.savez_compressed(temporary_npz, **{k: np.asarray(v) for k, v in arrays.items()})
    temporary_json.write_text(
        json.dumps({"format_version": 1, **dict(metadata)}, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    temporary_npz.replace(npz_path)
    temporary_json.replace(json_path)
    return npz_path, json_path


def completed_artifact_matches(prefix: Path | str, fingerprint: str) -> bool:
    base = Path(prefix).expanduser()
    npz_path, json_path = base.with_suffix(".npz"), base.with_suffix(".json")
    if not npz_path.is_file() or not json_path.is_file():
        return False
    try:
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return metadata.get("status") == "complete" and metadata.get("fingerprint") == fingerprint


def config_as_metadata(config: ArticleReproductionConfig) -> dict[str, Any]:
    result = asdict(config)
    result["output_dir"] = str(result["output_dir"])
    return result


__all__ = [
    "ARTICLE_CASES",
    "ARTICLE_DUST_MASS_KG",
    "ARTICLE_DUST_RADIUS_M",
    "ARTICLE_DUST_Z_OVER_LAMBDA",
    "ARTICLE_ELECTRON_TEMPERATURE_K",
    "ARTICLE_ION_MASS_KG",
    "ARTICLE_ION_TEMPERATURE_K",
    "ARTICLE_PRESSURE_PA",
    "ARTICLE_REFERENCE_DENSITY_M3",
    "ArticleCase",
    "ArticleReproductionConfig",
    "PaperElectricField",
    "article_drift_mach",
    "article_sound_speed_m_s",
    "artifact_fingerprint",
    "completed_artifact_matches",
    "config_as_metadata",
    "fit_required_field",
    "normalized_force_imbalance",
    "paper_z_m",
    "required_field_V_m",
    "save_iteration_artifact",
    "simulation_z_m",
    "summarize_production_result",
]
