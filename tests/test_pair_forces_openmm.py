import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")
import jax.numpy as jnp

openmm = pytest.importorskip("openmm")
from openmm import unit

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from opendust_jax.pair_forces import softened_yukawa_forces


def openmm_forces(
    positions: np.ndarray,
    prefactor: float,
    screening_length: float,
    softening: float,
) -> np.ndarray:
    system = openmm.System()
    force = openmm.CustomNonbondedForce(
        "prefactor*exp(-r/screening_length)/(r+softening)"
    )
    force.addGlobalParameter("prefactor", prefactor)
    force.addGlobalParameter("screening_length", screening_length)
    force.addGlobalParameter("softening", softening)
    for _ in positions:
        system.addParticle(1.0)
        force.addParticle([])
    system.addForce(force)

    integrator = openmm.VerletIntegrator(1.0)
    context = openmm.Context(system, integrator)
    context.setPositions(positions)
    state = context.getState(getForces=True)
    return np.asarray(
        state.getForces(asNumpy=True).value_in_unit(
            unit.kilojoule_per_mole / unit.nanometer
        )
    )


def test_jax_matches_isolated_openmm_force():
    rng = np.random.default_rng(42)
    positions = rng.normal(size=(97, 3)).astype(np.float32)
    prefactor = 1.7
    screening_length = 0.9
    softening = 1.0e-3

    expected = openmm_forces(
        positions, prefactor, screening_length, softening
    )
    actual = np.asarray(
        softened_yukawa_forces(
            jnp.asarray(positions),
            prefactor,
            screening_length,
            softening,
            block_size=32,
        )
    )

    difference = actual - expected
    relative_l2 = np.linalg.norm(difference) / np.linalg.norm(expected)
    max_absolute = np.max(np.abs(difference))
    nonzero = np.abs(expected) > np.finfo(np.float32).eps
    max_relative = np.max(np.abs(difference[nonzero] / expected[nonzero]))
    print(
        f"relative_l2={relative_l2:.8e} "
        f"max_absolute={max_absolute:.8e} max_relative={max_relative:.8e}"
    )

    assert np.all(np.isfinite(actual))
    assert relative_l2 <= 5.0e-5
