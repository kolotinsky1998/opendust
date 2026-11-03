"""Configuration for OpenDust paper Figure 11."""

from __future__ import annotations

from common import (
    CaseSetup,
    DEFAULT_BACKGROUND_SAMPLES,
    ELEMENTARY_CHARGE_C,
    averaging_start,
    production_or_override,
)


MACH_VALUES = (0.29940, 0.40129, 0.50033, 0.60133, 0.70098, 0.80019, 1.00091)
CASE_COUNT = len(MACH_VALUES)
FIXED_CHARGE_E = -40_939.491598


def build_case(args, results_dir):
    from opendust_jax import (
        DustParticleSI,
        MaxwellianPlasmaSI,
        PolynomialBackgroundConfig,
        SimulationConfigSI,
    )

    mach = MACH_VALUES[args.case_index]
    plasma = MaxwellianPlasmaSI(290.11, 29_011.0, 1.0e14, mach, 6.6335209e-26)
    steps = production_or_override(args.steps, 50_000)
    simulation = SimulationConfigSI(
        3.0 * plasma.screening_length_m,
        6.0 * plasma.screening_length_m,
        production_or_override(args.particles, 2**17),
        steps,
        1.0e-9,
        seed=args.seed,
        force_backend="cuda",
        step_chunk_size=args.chunk_size,
    )
    charge = FIXED_CHARGE_E * ELEMENTARY_CHARGE_C
    dust = (
        DustParticleSI(
            (0.0, 0.0, -0.5 * plasma.screening_length_m),
            12.0e-6,
            initial_charge_C=charge,
            charge_mode="fixed",
        ),
        DustParticleSI(
            (0.5 * plasma.screening_length_m, 0.0, 0.5 * plasma.screening_length_m),
            12.0e-6,
            initial_charge_C=charge,
            charge_mode="fixed",
        ),
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
        dust,
        background,
        {"case_index": args.case_index, "mach": mach},
        averaging_start(10_000, steps),
    )
