"""JAX-based validation utilities for OpenDust FMM experiments."""

from opendust_jax.direct import direct_coulomb_field, direct_coulomb_forces
from opendust_jax.fmm_poisson import (
    build_fmm_tree,
    fmm_coulomb_field,
    fmm_coulomb_forces,
)
from opendust_jax.geometry import CylinderDomain, points_inside_cylinder, sample_uniform_cylinder
from opendust_jax.validation import ValidationMetrics, compute_force_metrics

__all__ = [
    "CylinderDomain",
    "ValidationMetrics",
    "build_fmm_tree",
    "compute_force_metrics",
    "direct_coulomb_field",
    "direct_coulomb_forces",
    "fmm_coulomb_field",
    "fmm_coulomb_forces",
    "points_inside_cylinder",
    "sample_uniform_cylinder",
]
