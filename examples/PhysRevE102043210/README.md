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
   weighting inside the strict admissible field region, evaluates Eq. (15),
   prepares a new background, and runs production.
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

Every production stage uses the outer ion--ion force step
`dt = 0.1 / omega_pi`. The number of outer steps is rounded up so that the run
spans at least `83 * (2*pi/omega_pi)` (5,216 steps with the article parameters),
and the default `16.6 tau_i` burn-in begins at step 1,044. The expensive plane
diagnostic is sampled every 10 outer steps, preserving approximately the old
physical sampling cadence. `CollisionSubcyclingConfig()` resolves the bounded
`2 * sound_speed + 6 * ion_thermal_speed` majorant inside each outer step,
while `ChargingSubcyclingConfig()` chooses the still smaller charging step from
the 10% dust-radius displacement criterion.

Presheath preparation uses its own collision-limited step. Both preparation
and production use the `2 * sound_speed + 6 * ion_thermal_speed` collision
majorant independently of the inlet Mach number; only production places that
operator on an inner grid below its `0.1 / omega_pi` outer step.
Axial density and flux are sampled once per ion plasma period and convergence
is evaluated from adjacent time-window averages.

Each stage prints and records separate wall-clock durations for presheath
background preparation/loading, production (including JIT compilation),
postprocessing, and the complete stage. CUDA work is synchronized before the
production timer stops. The background timing also reports whether a compatible
cache was reused. A branch array task still runs iterations 2--5 sequentially,
so it performs one million outer pair-force updates per case.

Both the raw unweighted fit and the applied constrained fit are written to the
iteration JSON. Linear fits enforce nonnegative paper-coordinate slope;
quadratic fits enforce positive curvature and the strict vertex bounds used by
`QuadraticElectricFieldSI`. If the bounded optimizer fails, the workflow uses
its admissible projected starting point instead of aborting the branch.

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

## Short production-chain benchmark

`benchmark_chain.py` is a reduced-duration performance workload for testing
optimizations without running a complete Matthews-chain iteration.  It keeps
the Case-I production geometry, 65,536 ions, seven self-consistently charged
dust grains, prepared spline background, `0.1 / omega_pi` outer step,
collision and charging subcycling, and the `81x161` plane diagnostics.  The
diagnostics start at step 1 so that the short workload samples them at the same
10-step cadence as the post-burn-in part of production.

The default is one 32-step compile warm-up followed by one identical measured
run.  On a node where the 5,216-step stage takes about 90 minutes, the two runs
together should take approximately one minute.  Run it after the Case-I
constant background has been prepared:

```bash
cd /home/avtimofeev/developing_opendust/opendust/examples/PhysRevE102043210
sbatch slurm_chain_benchmark.sh
```

The measured time, resolved substep counts, operation counts, collision
statistics, and a linear full-stage projection are written to
`results/chain_benchmark/chain_benchmark_default.json`. Useful controlled variants
include:

```bash
# Isolate the cost of plane diagnostics.
sbatch slurm_chain_benchmark.sh --label no-diagnostics --no-diagnostics

# Explore a charging-grid change without editing the article workflow.
sbatch slurm_chain_benchmark.sh --label charge-64 --charging-substeps 64

# Longer sample or several measured repetitions after one warm-up.
sbatch slurm_chain_benchmark.sh --label longer --steps 64 --repeat 3
```

Use the same command and background before and after a code optimization.  The
warm-up compiles the exact measured scan shape and is excluded from the
reported median.  Pass `--warmup 0` only when end-to-end JIT time itself is the
quantity being tested.
