# Reproduction of Phys. Rev. E 102, 043210

This directory runs the three article cases and produces Figures 3--6 plus the
Figure 7b force-convergence panel.  The physical configuration is kept in
`article_config.json`; full production requires the compiled CUDA FFI backend
and one NVIDIA GPU per Slurm array task.

The workflow is intentionally staged:

1. Each constant-field array task prepares a compatible stationary presheath
   background and runs iteration 1.
2. The linear and quadratic arrays independently run iterations 2--5 in order.
   Each iteration fits the preceding seven force-balance values without
   weighting, evaluates Eq. (15), prepares a new background, and runs production.
3. Postprocessing writes PNG, PDF, CSV, and `validation_report.json`.

Submit the complete dependency graph with:

```bash
module load cuda/12.9
conda activate opendust-jax
cd /home/avtimofeev/developing_opendust/opendust/examples/PhysRevE102043210
bash submit_slurm.sh
```

The submitter pins the active Python executable, passes the absolute example
directory into every batch job, and cancels an incomplete dependency graph if
a later `sbatch` call fails. GPU jobs stop immediately unless JAX sees a GPU and
the OpenDust CUDA FFI library is available.

Completed stages are skipped only when their input fingerprint matches.  Delete
or move a stage pair (`iteration_XX.npz` and `.json`) to force recomputation.
Background caches carry their own plasma/geometry/cross-section fingerprint.

For a local functional check (small JAX reference problem, not a physical
reproduction), run:

```bash
JAX_ENABLE_X64=1 python3 workflow.py --stage smoke
```

The reported outer step resolves the ion plasma frequency and updates the
ion-ion force once. `CollisionSubcyclingConfig()` resolves the null-collision
majorant inside it, while `ChargingSubcyclingConfig()` chooses the still smaller
charging step from the 10% dust-radius displacement criterion. Diagnostics and
the full duration (83(2\pi/\omega_{pi})) remain on the outer-step time grid.

The implementation deliberately uses OpenDust's physical absorption radius and
the existing orbit+collection force.  It does not implement the DRIAD critical
radius or Maxwell-stress tensor; this difference is recorded in every stage and
in the validation report.

## Standalone Figure 3 workflow

Figure 3 can be reproduced independently of the dust-chain iterations.  It
runs three dust-free Case-II presheaths using the constant, fifth-iteration
linear, and fifth-iteration quadratic fields printed in Table I, then overlays
the resulting density and drift profiles with an approximate digitization of
the published curves:

```bash
cd /home/avtimofeev/developing_opendust/opendust/examples/PhysRevE102043210
sbatch slurm_figure3.sh
```

Outputs are written to `results/figure3_table_I/`, including
`figure3_comparison.png`, `figure3_comparison.pdf`, the comparison CSV, and a
JSON file with density/velocity RMSE.  Compatible stationary-background files
are reused on restart.  The bundled article curves are explicitly approximate
pixel digitizations, not original author data.

A short CPU/JAX functional check is available with:

```bash
JAX_ENABLE_X64=1 python3 figure3_workflow.py --smoke --profile all
```
