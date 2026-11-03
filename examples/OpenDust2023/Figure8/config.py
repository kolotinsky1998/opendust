"""Figure 8 inputs copied literally from legacy Piel2017/Figure5."""

from __future__ import annotations

from common import (
    CaseSetup,
    DEFAULT_BACKGROUND_SAMPLES,
    ELEMENTARY_CHARGE_C,
    averaging_start,
    production_or_override,
)


MACH_VALUES = (
    0.19679,
    0.30120,
    0.40161,
    0.49799,
    0.60241,
    0.69880,
    0.80321,
    0.89960,
    0.99598,
    1.19679,
    1.49799,
    2.00000,
)
CASE_COUNT = len(MACH_VALUES)
INITIAL_CHARGE_E = -392_500.0


def build_case(args, results_dir):
    from opendust_jax import (
        DustParticleSI,
        MaxwellianPlasmaSI,
        PolynomialBackgroundConfig,
        SimulationConfigSI,
    )

    mach = MACH_VALUES[args.case_index]
    plasma = MaxwellianPlasmaSI(290.11, 29_011.0, 1.0e14, mach, 6.6335209e-26)
    steps = production_or_override(args.steps, 5000)
    simulation = SimulationConfigSI(
        3.0 * plasma.screening_length_m,
        6.0 * plasma.screening_length_m,
        production_or_override(args.particles, 2**16),
        steps,
        3.5148240854e-9,
        seed=args.seed,
        force_backend="cuda",
        step_chunk_size=args.chunk_size,
    )
    dust = DustParticleSI(
        (0.0, 0.0, 0.0),
        58.8e-6,
        initial_charge_C=INITIAL_CHARGE_E * ELEMENTARY_CHARGE_C,
        charge_mode="oml",
    )
    background = PolynomialBackgroundConfig(
        seed=args.seed,
        sample_count=production_or_override(
            args.background_samples, DEFAULT_BACKGROUND_SAMPLES
        ),
        cache_dir=results_dir / "background_cache",
    )
    return CaseSetup(
        plasma,
        simulation,
        (dust,),
        background,
        {"case_index": args.case_index, "mach": mach},
        averaging_start(3000, steps),
    )
