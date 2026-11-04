"""Prepare and run a non-uniform argon presheath case."""

from pathlib import Path

from opendust_jax import (
    ArgonPresheathPlasmaSI,
    DustParticleSI,
    OpenDustJAX,
    OutputConfig,
    PresheathEquilibrationConfig,
    QuadraticElectricFieldSI,
    SimulationConfigSI,
    prepare_presheath_background,
)


HERE = Path(__file__).resolve().parent

plasma = ArgonPresheathPlasmaSI(
    neutral_temperature_K=290.0,
    electron_temperature_K=58_022.0,
    reference_density_m3=1.0e14,
    pressure_Pa=18.131842,
    inlet_mach=0.30,
    electric_field=QuadraticElectricFieldSI(
        E0_V_m=2614.9630526115625,
        alpha_V_m2=40_000.0,
        beta_V_m3=-40_000_000.0,
    ),
)
lambda_de = plasma.screening_length_m
common = dict(
    radius_m=1.25 * lambda_de,
    height_m=5.0 * lambda_de,
    particle_count=2**16,
    dt_s=1.0e-10,
    seed=7,
    force_backend="cuda",
    step_chunk_size=100,
)

preparation = SimulationConfigSI(steps=0, **common)
background = prepare_presheath_background(
    plasma,
    preparation,
    PresheathEquilibrationConfig(
        output_prefix=HERE / "results" / "presheath",
    ),
)

production = SimulationConfigSI(steps=10_000, **common)
dust = DustParticleSI(
    position_m=(0.0, 0.0, -lambda_de),
    radius_m=4.445e-6,
    initial_charge_C=-2.5e-15,
    charge_mode="self_consistent",
)
result = OpenDustJAX(
    plasma,
    production,
    [dust],
    background=background,
    output=OutputConfig(checkpoint_path=HERE / "results" / "restart.npz"),
).run()
print(result.collision_statistics)
