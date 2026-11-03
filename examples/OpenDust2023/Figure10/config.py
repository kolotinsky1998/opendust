"""Figure 10 inputs copied literally from legacy Patacchini2008/Figure3."""

from __future__ import annotations

from common import (
    CaseSetup,
    DEFAULT_BACKGROUND_SAMPLES,
    averaging_start,
    production_or_override,
)


COLLISION_FREQUENCIES_S = (
    1.57319303e04,
    1.57319303e05,
    4.87689840e05,
    1.66758461e06,
    4.93982612e06,
    1.64084033e07,
    5.00275384e07,
    1.65562835e08,
    4.89121446e08,
    1.65767350e09,
    3.22479401e09,
    8.05204244e09,
    1.61113688e10,
)
ELECTRIC_FIELDS_V_M = (
    2.58520000e00,
    2.58520000e01,
    8.01412000e01,
    2.74031200e02,
    8.11752800e02,
    2.69636360e03,
    8.22093600e03,
    2.72066448e04,
    8.03764532e04,
    2.72402524e05,
    5.29924637e05,
    1.32317775e06,
    2.64755244e06,
)
CASE_COUNT = len(COLLISION_FREQUENCIES_S)


def build_case(args, results_dir):
    from opendust_jax import (
        DustParticleSI,
        FieldDrivenPlasmaSI,
        PolynomialBackgroundConfig,
        SimulationConfigSI,
    )

    collision_frequency = COLLISION_FREQUENCIES_S[args.case_index]
    electric_field = ELECTRIC_FIELDS_V_M[args.case_index]
    plasma = FieldDrivenPlasmaSI(
        300.0,
        30_000.0,
        3.57167962497e15,
        electric_field,
        collision_frequency,
        1.673557e-27,
    )
    steps = production_or_override(args.steps, 200_000)
    simulation = SimulationConfigSI(
        3.0 * plasma.screening_length_m,
        6.0 * plasma.screening_length_m,
        production_or_override(args.particles, 2**17),
        steps,
        1.0e-11,
        seed=args.seed,
        force_backend="cuda",
        step_chunk_size=args.chunk_size,
    )
    dust = DustParticleSI(
        (0.0, 0.0, -plasma.screening_length_m),
        1.0e-5,
        initial_charge_C=0.0,
        charge_mode="self_consistent",
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
        {
            "case_index": args.case_index,
            "collision_frequency_s": collision_frequency,
            "electric_field_V_m": electric_field,
        },
        averaging_start(150_000, steps),
    )
