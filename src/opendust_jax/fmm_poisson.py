"""Coulomb/Laplace FMM wrappers for the first JAX OpenDust validation milestone."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import jax
import jax.numpy as jnp

from opendust_jax.direct import _validate_inputs


def _ensure_local_jaxfmm_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    local_jaxfmm = repo_root / "jaxfmm-main"
    if local_jaxfmm.exists():
        path = str(local_jaxfmm)
        if path not in sys.path:
            sys.path.insert(0, path)


def _import_jaxfmm() -> tuple[Any, Any]:
    _ensure_local_jaxfmm_on_path()
    try:
        from jaxfmm import eval_potential, gen_hierarchy
    except ImportError as exc:
        raise ImportError(
            "Could not import jaxfmm. Install jaxfmm or keep the local "
            "'jaxfmm-main' directory at the repository root."
        ) from exc
    return eval_potential, gen_hierarchy


def build_fmm_tree(
    positions: jax.Array,
    n_max: int = 128,
    theta: float = 0.77,
    p: int = 4,
) -> dict[str, Any]:
    """Build a jaxFMM hierarchy for fixed source/evaluation positions."""

    positions = jnp.asarray(positions)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3).")
    if n_max <= 0:
        raise ValueError("n_max must be positive.")
    if theta <= 0:
        raise ValueError("theta must be positive.")
    if p < 0:
        raise ValueError("p must be non-negative.")

    _, gen_hierarchy = _import_jaxfmm()
    return gen_hierarchy(positions, N_max=int(n_max), theta=float(theta), p=int(p))


def fmm_coulomb_field(
    positions: jax.Array,
    charges: jax.Array,
    eps0: float = 8.85418781762039e-12,
    p: int = 4,
    theta: float = 0.77,
    n_max: int = 128,
    tree: dict[str, Any] | None = None,
) -> jax.Array:
    """Evaluate the free-space Coulomb electric field using jaxFMM.

    jaxFMM evaluates the field of the Laplace kernel 1 / (4*pi*r). Dividing by
    eps0 converts this to the physical electric field in SI units.
    """

    positions, charges = _validate_inputs(positions, charges)
    if eps0 <= 0:
        raise ValueError("eps0 must be positive.")

    eval_potential, _ = _import_jaxfmm()
    if tree is None:
        tree = build_fmm_tree(positions, n_max=n_max, theta=theta, p=p)
    return eval_potential(charges, **tree, field=True) / eps0


def fmm_coulomb_forces(
    positions: jax.Array,
    charges: jax.Array,
    eps0: float = 8.85418781762039e-12,
    p: int = 4,
    theta: float = 0.77,
    n_max: int = 128,
    tree: dict[str, Any] | None = None,
) -> jax.Array:
    """Evaluate Coulomb forces on each particle with jaxFMM."""

    positions, charges = _validate_inputs(positions, charges)
    field = fmm_coulomb_field(
        positions,
        charges,
        eps0=eps0,
        p=p,
        theta=theta,
        n_max=n_max,
        tree=tree,
    )
    return charges[:, None] * field
