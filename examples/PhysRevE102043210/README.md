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
bash submit_slurm.sh
```

Completed stages are skipped only when their input fingerprint matches.  Delete
or move a stage pair (`iteration_XX.npz` and `.json`) to force recomputation.
Background caches carry their own plasma/geometry/cross-section fingerprint.

For a local functional check (small JAX reference problem, not a physical
reproduction), run:

```bash
JAX_ENABLE_X64=1 python3 workflow.py --stage smoke
```

The reported main step resolves the ion plasma frequency and null-collision
majorant.  `ChargingSubcyclingConfig()` automatically chooses the smaller
charging step from the 10% dust-radius displacement criterion.  Diagnostics and
the reported 83 plasma-period duration remain on the main-step time grid.

The implementation deliberately uses OpenDust's physical absorption radius and
the existing orbit+collection force.  It does not implement the DRIAD critical
radius or Maxwell-stress tensor; this difference is recorded in every stage and
in the validation report.
