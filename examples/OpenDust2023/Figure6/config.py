"""Configuration for OpenDust paper Figure 6."""

from __future__ import annotations

from common import (
    CaseSetup,
    DEFAULT_BACKGROUND_SAMPLES,
    averaging_start,
    production_or_override,
)


CHARGE_MODES = ("self_consistent", "oml")
CASE_COUNT = len(CHARGE_MODES)


def build_case(args, results_dir):
    from opendust_jax import (
        DustParticleSI,
        MaxwellianPlasmaSI,
        PolynomialBackgroundConfig,
        SimulationConfigSI,
    )

    mode = CHARGE_MODES[args.case_index]
    plasma = MaxwellianPlasmaSI(290.11, 29_011.0, 1.0e14, 1.0, 6.6335209e-26)
    steps = production_or_override(args.steps, 3000)
    simulation = SimulationConfigSI(
        radius_m=3.0 * plasma.screening_length_m,
        height_m=6.0 * plasma.screening_length_m,
        particle_count=production_or_override(args.particles, 2**16),
        steps=steps,
        dt_s=3.5148240854e-9,
        seed=args.seed,
        force_backend="cuda",
        step_chunk_size=args.chunk_size,
    )
    dust = DustParticleSI(
        (0.0, 0.0, -plasma.screening_length_m),
        58.8e-6,
        initial_charge_C=0.0,
        charge_mode=mode,
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
        {"case_index": args.case_index, "charge_mode": mode},
        averaging_start(1000, steps),
    )
