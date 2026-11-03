# OpenDust 2023 figure reproductions

These single-GPU JAX/CUDA examples reproduce the OpenDust-only data for paper
Figures 6, 7, 8, 10, and 11. No literature reference curves are included.
The numerical inputs intentionally follow the legacy scripts on `origin/main`.
Consequently, Figure 8 uses evolving OML charge rather than the fixed-charge
wording in the paper, and Figure 10 uses `R=3 r_D` with the dust at `z=-r_D`.

| Folder | Calculation | Slurm cases |
| --- | --- | ---: |
| `Figure6` | OML versus self-consistent charge history | 2 |
| `Figure7` | Ion-drag time history | 1 |
| `Figure8` | Orbital ion drag versus Mach number | 12 |
| `Figure10` | Collisional ion drag and charge | 13 |
| `Figure11` | Two-particle restoring force | 7 |

The production scripts expect the `opendust-jax` conda environment to contain
JAX with CUDA support, NumPy, SciPy, and Matplotlib. Build the CUDA FFI library
for the target GPU before submitting any example:

```bash
module load cuda/12.9
conda activate opendust-jax
python performance/JAX/build_cuda_pair_forces.py --cuda-arch 70
```

Install the plotting dependency if it is not already present with
`conda install -c conda-forge matplotlib`.

Use `--cuda-arch 80` for A100. Submit an example from its directory with
`sbatch submit.sbatch`. Each array element uses one GPU. Raw case histories,
metadata, aggregate CSV, and final PNG/PDF figures are written below that
example's ignored `results/` directory.
