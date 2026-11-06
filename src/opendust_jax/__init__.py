"""JAX/CUDA implementation and validation utilities for OpenDust."""

from opendust_jax.direct import (
    direct_coulomb_field,
    direct_coulomb_forces,
    direct_yukawa_field,
    direct_yukawa_forces,
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
from opendust_jax.collisions import (
    ArgonNullCollisionModel,
    CollisionModel,
    LegacyChargeExchangeCollision,
    load_argon_cross_sections,
)
from opendust_jax.diagnostics import (
    plane_cic_density_snapshot_m3,
    plane_potential_snapshot_V,
)
from opendust_jax.models import (
    ArgonPresheathPlasmaSI,
    ChargingSubcyclingConfig,
    CollisionSubcyclingConfig,
    CollisionStatistics,
    DustParticleSI,
    FieldDrivenPlasmaSI,
    MaxwellianPlasmaSI,
    OutputConfig,
    PlaneDiagnosticConfig,
    PolynomialBackgroundConfig,
    PresheathEquilibrationConfig,
    QuadraticElectricFieldSI,
    SimulationConfigSI,
    SimulationResult,
    SimulationState,
)
from opendust_jax.timestepping import TimeStepRecommendation, recommend_main_timestep
from opendust_jax.presheath import (
    CylindricalSplineBackground,
    load_presheath_background,
    prepare_presheath_background,
    save_presheath_background,
)
from opendust_jax.simulation import OpenDustJAX
from opendust_jax.validation import ValidationMetrics, compute_force_metrics
from opendust_jax.article_reproduction import (
    ARTICLE_CASES,
    ARTICLE_DUST_Z_OVER_LAMBDA,
    ArticleCase,
    ArticleReproductionConfig,
    PaperElectricField,
    article_drift_mach,
    fit_required_field,
    normalized_force_imbalance,
    paper_z_m,
    required_field_V_m,
    simulation_z_m,
)

__all__ = [
    "CylinderDomain",
    "ArgonNullCollisionModel",
    "ArgonPresheathPlasmaSI",
    "ARTICLE_CASES",
    "ARTICLE_DUST_Z_OVER_LAMBDA",
    "ArticleCase",
    "ArticleReproductionConfig",
    "ChargingSubcyclingConfig",
    "CollisionSubcyclingConfig",
    "CollisionStatistics",
    "CollisionModel",
    "CylindricalSplineBackground",
    "DustParticleSI",
    "FieldDrivenPlasmaSI",
    "LegacyChargeExchangeCollision",
    "MaxwellianPlasmaSI",
    "OpenDustJAX",
    "OutputConfig",
    "PaperElectricField",
    "PlaneDiagnosticConfig",
    "PolynomialBackground",
    "PolynomialBackgroundConfig",
    "PresheathEquilibrationConfig",
    "QuadraticElectricFieldSI",
    "SimulationConfigSI",
    "SimulationResult",
    "SimulationState",
    "TimeStepRecommendation",
    "ValidationMetrics",
    "compute_force_metrics",
    "direct_coulomb_field",
    "direct_coulomb_forces",
    "direct_yukawa_field",
    "direct_yukawa_forces",
    "points_inside_cylinder",
    "sample_uniform_cylinder",
    "softened_yukawa_forces",
    "softened_yukawa_forces_2d",
    "softened_yukawa_forces_cuda",
    "cuda_pair_forces_available",
    "external_forces_cuda",
    "build_polynomial_background",
    "article_drift_mach",
    "fit_required_field",
    "load_checkpoint",
    "load_argon_cross_sections",
    "load_presheath_background",
    "normalized_force_imbalance",
    "paper_z_m",
    "plane_cic_density_snapshot_m3",
    "plane_potential_snapshot_V",
    "save_checkpoint",
    "prepare_presheath_background",
    "save_presheath_background",
    "recommend_main_timestep",
    "required_field_V_m",
    "simulation_z_m",
    "softened_yukawa_forces_cuda_with_stats",
]
