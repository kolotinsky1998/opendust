"""JAX-based validation utilities for OpenDust FMM experiments."""

from opendust_jax.direct import (
    direct_coulomb_field,
    direct_coulomb_forces,
    direct_yukawa_field,
    direct_yukawa_forces,
)
from opendust_jax.fmm_poisson import (
    build_fmm_tree,
    fmm_coulomb_field,
    fmm_coulomb_forces,
)
from opendust_jax.geometry import CylinderDomain, points_inside_cylinder, sample_uniform_cylinder
from opendust_jax.pair_forces import (
    softened_yukawa_forces,
    softened_yukawa_forces_2d,
)
from opendust_jax.cuda_pair_forces import (
    cuda_pair_forces_available,
    external_forces_cuda,
    softened_yukawa_forces_cuda,
    softened_yukawa_forces_cuda_with_stats,
)
from opendust_jax.background import PolynomialBackground, build_polynomial_background
from opendust_jax.checkpoint import load_checkpoint, save_checkpoint
from opendust_jax.collisions import CollisionModel, LegacyChargeExchangeCollision
from opendust_jax.diagnostics import (
    plane_cic_density_snapshot_m3,
    plane_potential_snapshot_V,
)
from opendust_jax.models import (
    DustParticleSI,
    FieldDrivenPlasmaSI,
    MaxwellianPlasmaSI,
    OutputConfig,
    PlaneDiagnosticConfig,
    PolynomialBackgroundConfig,
    SimulationConfigSI,
    SimulationResult,
    SimulationState,
)
from opendust_jax.simulation import OpenDustJAX
from opendust_jax.validation import ValidationMetrics, compute_force_metrics
from opendust_jax.yukawa_exponential_tables import (
    YukawaExponentialM2LTable,
    apply_exponential_m2l_table,
    build_yukawa_m2l_exponential_table,
    exponential_m2l_operator_from_table,
    validate_yukawa_m2l_exponential_table,
)
from opendust_jax.yukawa_fmm import (
    build_yukawa_tree,
    yukawa_fmm_field,
    yukawa_fmm_forces,
)

__all__ = [
    "CylinderDomain",
    "CollisionModel",
    "DustParticleSI",
    "FieldDrivenPlasmaSI",
    "LegacyChargeExchangeCollision",
    "MaxwellianPlasmaSI",
    "OpenDustJAX",
    "OutputConfig",
    "PlaneDiagnosticConfig",
    "PolynomialBackground",
    "PolynomialBackgroundConfig",
    "SimulationConfigSI",
    "SimulationResult",
    "SimulationState",
    "ValidationMetrics",
    "YukawaExponentialM2LTable",
    "apply_exponential_m2l_table",
    "build_fmm_tree",
    "build_yukawa_m2l_exponential_table",
    "compute_force_metrics",
    "direct_coulomb_field",
    "direct_coulomb_forces",
    "direct_yukawa_field",
    "direct_yukawa_forces",
    "fmm_coulomb_field",
    "fmm_coulomb_forces",
    "build_yukawa_tree",
    "points_inside_cylinder",
    "sample_uniform_cylinder",
    "softened_yukawa_forces",
    "softened_yukawa_forces_2d",
    "softened_yukawa_forces_cuda",
    "cuda_pair_forces_available",
    "external_forces_cuda",
    "build_polynomial_background",
    "load_checkpoint",
    "plane_cic_density_snapshot_m3",
    "plane_potential_snapshot_V",
    "save_checkpoint",
    "softened_yukawa_forces_cuda_with_stats",
    "exponential_m2l_operator_from_table",
    "validate_yukawa_m2l_exponential_table",
    "yukawa_fmm_field",
    "yukawa_fmm_forces",
]
