# Chain rotation experiment

This workflow performs the five-stage force-balance iteration for the measured
eight-particle chain. The first stage uses a constant field corresponding to
inlet Mach 0.5. Stages 2--5 use the constrained quadratic field fitted to the
previous stage's force-balance targets. The inlet Mach of those stages is
recomputed from the field at the upstream particle.

The default parameter grid is:

- electron temperature: 3, 4, and 5 eV;
- quasineutral density: 1e14, 1e15, and 1e16 m^-3;
- pressure: 11.5 Pa;
- dust diameter: 10.5 micrometres;
- dust material density: 1490 kg/m^3.

Only `particle_id`, `x_m`, `y_m`, and `z_m` are read from
`experiment2_marker13_mean_rotation_frame.csv`. The force columns in that file
are not simulation inputs.

## Inspect and test locally

Print the resolved geometry and superion count for all nine cases:

```bash
python workflow.py --describe-grid
```

Run a small CPU smoke calculation for one grid point:

```bash
JAX_PLATFORMS=cpu python workflow.py --case-index 0 --smoke
JAX_PLATFORMS=cpu python postprocess.py --smoke
```

The smoke mode still executes all five stages, but uses 64 superions, short
histories, and a small diagnostic grid. It checks workflow integration, not
physical convergence.

For temporary checks, `--output-dir /path/to/output` overrides the configured
results directory for both commands.

## Submit to Slurm

Activate the environment that contains the installed checkout and built CUDA
extension, then run:

```bash
./submit_slurm.sh
```

`slurm_array.sh` submits indices 0--8 to the `rocky` partition. Each index owns
one plasma-grid point and runs the constant stage followed by four quadratic
stages. The dependent CPU job runs postprocessing only after the complete array
succeeds. A failed index can be resubmitted directly; compatible finished stage
artifacts are skipped by fingerprint.

## Results

Each case is stored below `results/Te_<T>eV_n_1e<N>/`. Every stage has an
atomic NPZ/JSON pair, a restart checkpoint, a prepared-presheath cache, and an
XZ/YZ diagnostic field file. The NPZ histories include charge and all Cartesian
components of orbit, collection, dust--dust, external, and total forces.

`results/figures/` contains the common geometry, the nine-case summary CSV, and
one directory per plasma-grid point. Per-case products include density and
potential maps, background and central-axis profiles, charge and force
iterations, final force decomposition, quadratic-field fits, and final-stage
autocorrelations. Means and 95% intervals use the integrated autocorrelation
time; the tables flag charge SEM above 1% and detected nonstationarity.

The force model follows the current OpenDust implementation: physical dust
absorption radius plus orbit and collection forces. It does not replace these
with the DRIAD critical-radius or Maxwell-stress construction.
