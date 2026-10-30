# OpenDust
OpenDust: A fast GPU-accelerated code for calculation forces, acting on microparticles in a plasma flow
<img src="https://github.com/kolotinsky1998/opendust/blob/main/animation/animation.gif" width="600" height="450" />

Opendust is GPU-based, Python library for dusty plasmas. 

OpenDust aims to provide researchers both experimenters and theorists user-friendly and high-performance tool for calculation forces, acting on microparticles, and microparticles’ charges in a plasma flow. 

OpenDust performance originates from highly-optimized Cuda back-end and allows to perform self-consistent calculation of plasma flow around microparticles in seconds.

## Installation
OpenDust is installed using [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

1. Begin by installing the most recent 64 bit, Python 3.x version of Miniconda
2. For using OpenDust you need Nvidia GPU. Make sure you have installed modern GPU [drivers](https://www.nvidia.com/Download/index.aspx). CUDA itself will be provided by the `cudatoolkit` package when you install `opendust` in the next steps.
3. Open a command line terminal and type the following commands
```
conda create -n opendust
```
```
conda activate opendust
```
```
conda install -c kolotinsky -c conda-forge opendust
```
With recent conda versions (v4.8.4+), this will install a version of OpenDust compiled with the latest version of CUDA supported by your drivers. Alternatively you can request a version that is compiled for a specific CUDA version with the command
```
conda install -c kolotinsky -c conda-forge opendust cudatoolkit=10.0
```
where `10.0` should be replaced with the particular CUDA version you want to target.
##  Documentation
https://github.com/kolotinsky1998/opendust/tree/main/tutorial

##  How to cite
In case of usage OpenDust please cite the article:

Kolotinskii D., Timofeev A. OpenDust: A fast GPU-accelerated code for the calculation of forces acting on microparticles in a plasma flow //Computer Physics Communications. – 2023. – Т. 288. – С. 108746.

##  Author's contacts

If you have any questions, comments or offers related to the OpenDust package or the base article do not to hesitate to contact me!

## Experimental JAX/CUDA implementation

`opendust_jax.OpenDustJAX` is an independent SI-only implementation for one
NVIDIA GPU. It retains Piel's asymmetric ion-dust model and uses the custom
triangular CUDA pair kernel through JAX XLA FFI. Build the native library first:

```bash
python3 performance/JAX/build_cuda_pair_forces.py --cuda-arch 70  # V100
```

Minimal fixed-charge simulation:

```python
from opendust_jax import (
    DustParticleSI, MaxwellianPlasmaSI, OpenDustJAX,
    PolynomialBackgroundConfig, SimulationConfigSI,
)

plasma = MaxwellianPlasmaSI(
    ion_temperature_K=290.11,
    electron_temperature_K=29011.0,
    density_m3=1.0e14,
    mach=1.0,
    ion_mass_kg=6.6335209e-26,
)
lambda_e = plasma.screening_length_m
simulation = SimulationConfigSI(
    radius_m=3 * lambda_e,
    height_m=6 * lambda_e,
    particle_count=2**16,
    steps=3000,
    dt_s=3.5148e-10,
)
dust = DustParticleSI(
    position_m=(0.0, 0.0, -lambda_e),
    radius_m=58.8e-6,
    initial_charge_C=-392500 * 1.602176634e-19,
)
result = OpenDustJAX(
    plasma,
    simulation,
    [dust],
    background=PolynomialBackgroundConfig(cache_dir=".opendust_cache"),
).run()
```

All public inputs and results use metres, seconds, kilograms, coulombs, volts,
and newtons. See `performance/JAX/README.md` for CUDA build and benchmark
commands.

Email: kolotinskiy.da@phystech.edu

Research Gate: https://www.researchgate.net/profile/Daniil_Kolotinskii

Telegram: @daniil_kolotinskii
