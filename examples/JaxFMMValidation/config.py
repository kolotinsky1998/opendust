"""Default configuration for the Coulomb FMM validation example."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CoulombValidationConfig:
    n_particles: int = 4096
    radius_m: float = 5.0e-4
    height_m: float = 1.0e-3
    charge_c: float = 1.60217662e-19
    eps0: float = 8.85418781762039e-12
    seed: int = 124
    p: int = 4
    theta: float = 0.77
    n_max: int = 128
    direct_batch_size: int = 256
    acceptance_relative_l2: float = 1.0e-2
    output_dir: Path = Path(__file__).resolve().parent / "output"


DEFAULT_CONFIG = CoulombValidationConfig()


@dataclass(frozen=True)
class YukawaValidationConfig(CoulombValidationConfig):
    debye_radius_factors: tuple[float, ...] = (0.5, 1.0, 2.0)
    output_dir: Path = Path(__file__).resolve().parent / "output_yukawa"


DEFAULT_YUKAWA_CONFIG = YukawaValidationConfig()
