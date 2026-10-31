# JAX direct pair-force benchmark

This benchmark compares the exact softened-Yukawa pair force implemented in
JAX/XLA with an isolated OpenMM `CustomNonbondedForce`. It intentionally does
not include integration, cutoffs, neighbor lists, FMM, collisions, or boundary
conditions.

Run it on one NVIDIA A100 from the repository root:

```bash
python3 performance/JAX/benchmark_pair_forces.py
```

The default run evaluates `N = 2^15, 2^16, 2^17, 2^18`, JAX block sizes 32,
64, 128, and 256, three warmups, and ten measured calls. The script requires a
CUDA-enabled JAX installation and the OpenMM CUDA platform in single precision.

Results are written to:

- `performance/JAX/results/pair_forces_results.csv`
- `performance/JAX/results/pair_forces_report.md`

Both implementations include host-to-device transfer, force evaluation,
explicit synchronization, and transfer of the force array back to the host in
each measured call. JAX compilation, OpenMM context creation, and coordinate
generation are reported separately and excluded from median execution time.

For a quick non-production smoke run, reduce the sizes and repetitions:

```bash
python3 performance/JAX/benchmark_pair_forces.py \
  --powers 10 11 --block-sizes 32 64 --warmup 1 --repeat 2
```

Use `--allow-non-cuda` only for diagnostic runs. Such results must not be used
for the JAX/OpenMM performance conclusion.

## Compare 1D and 2D tiling

The original backend evaluates one target tile against all sources. Pass
`--source-block-sizes` to additionally benchmark the bounded 2D backend for
every requested target/source tile combination:

```bash
python3 performance/JAX/benchmark_pair_forces.py \
  --powers 15 16 17 18 \
  --block-sizes 256 \
  --source-block-sizes 128 256 512 1024 \
  --warmup 3 \
  --repeat 10
```

The CSV identifies the implementations as `jax-1d` and `jax-2d`, and records
both target and source block sizes. Omitting `--source-block-sizes` preserves
the original benchmark behavior.

## Warp-tiled CUDA backend through JAX FFI

The optional `jax-cuda` backend evaluates every unordered pair once using
triangular `32x32` warp tiles. It is a CUDA implementation exposed as a JAX
operation through XLA FFI; positions and forces remain device arrays and the
operation can be used inside `jax.jit`.

Build it for a V100 (`sm_70`) in the same environment that contains JAX:

```bash
python3 performance/JAX/build_cuda_pair_forces.py --cuda-arch 70
```

The build needs `nvcc` from a CUDA 12 toolkit. On a module-based cluster, load
the matching CUDA toolkit before building. If the environment only contains
JAX's CUDA runtime wheels, install `cuda-nvcc` from NVIDIA's conda channel or
ask the cluster for its CUDA toolkit module. For A100 use `--cuda-arch 80`.

Run the same correctness check and timing comparison for OpenMM, the best
one-dimensional XLA tile, and the CUDA backend:

```bash
python3 performance/JAX/benchmark_pair_forces.py \
  --cuda-backend \
  --powers 15 16 17 18 \
  --block-sizes 256 \
  --warmup 3 \
  --repeat 10
```

The CUDA implementation currently accumulates tile results with FP32 atomic
adds. It does not implement `jax.grad`, `vmap`, multi-GPU sharding, cutoff, or
neighbor lists. The benchmark validates relative L2 error against OpenMM
before starting the large runs.

The same shared library now also contains the fused non-pair kernel used by
`OpenDustJAX`: bare ion<-dust Coulomb forces, the polynomial confining field,
the homogeneous driving field, and screened dust<-ion orbit-force reductions.
Rebuild the library after pulling these changes; an older `.so` does not export
the additional FFI target.

## Complete simulation benchmark

After building the CUDA library, benchmark complete integration steps with:

```bash
python3 performance/JAX/benchmark_simulation.py \
  --powers 15 16 17 18 --steps 100 --chunk-size 100 \
  --warmup 1 --repeat 5
```

The result is saved to `performance/JAX/results/simulation_results.csv`.
Compilation and initialization occur during the warmup and are excluded from
the measured repetitions. The existing pair-force benchmark remains the
like-for-like OpenMM comparison; the full benchmark measures the additional
integration, boundary, charging, and dust-force overhead.

## Piel2017 Figure 1 controlled comparison

The published `Figure1/consistent` case can be run through both the new CUDA
backend and legacy OpenMM implementation with identical initial ions and an
identical deterministic polynomial background:

The legacy side additionally requires `unyt` for its OpenMM unit conversions:

```bash
conda install -c conda-forge unyt
```

```bash
env JAX_ENABLE_X64=1 python3 -u \
  performance/JAX/benchmark_piel2017_figure1_consistent.py \
  --backend all
```

Defaults reproduce the example parameters: 65,536 superions, 3,000 steps,
`dt=3.5148240854e-9 s`, one initially neutral self-consistently charged dust
particle, and the full legacy-equivalent background source count. For a quick
pipeline check use `--particles 1024 --steps 20 --average-start-step 1
--background-samples 4096`; those reduced results are not physically
representative.

The `all` mode runs preparation, JAX, OpenMM, and comparison in separate
processes so their GPU contexts do not compete for memory. Results are placed
under `performance/JAX/results/piel2017_figure1_consistent/`, including raw NPZ
time series, per-backend timing JSON, `comparison.json`, and `comparison.md`.
Long-time charge and force means are compared after step 1001. Pointwise
trajectories are intentionally not treated as an error metric because the two
engines use different random streams after ion reinjection and the JAX version
contains the documented charging/indexing corrections.

## Piel2017 run with averaged plane diagnostics

Run the same JAX/CUDA physical case while accumulating electric potential and
CIC ion density in the `y=0` plane:

```bash
env JAX_ENABLE_X64=1 python3 -u \
  performance/JAX/run_piel2017_with_plane_diagnostics.py
```

The default `121x121` grid covers `x=[-R,R]` and `z=[-H/2,H/2]`. Snapshots are
sampled at steps 1000, 1100, ..., 3000 and averaged on the GPU using FP64
temporal accumulators. Density uses a slab of thickness `0.2 r_D`. Configure
these choices with `--grid-nx`, `--grid-nz`, `--diagnostic-start-step`,
`--diagnostic-interval`, and `--slice-thickness-debye`.

The final averaged arrays and their SI units are written to
`diagnostic_run/piel2017_plane_fields.npz` and `.json`. The NPZ contains
`x_m`, `z_m`, `potential_V`, `ion_density_m3`, `sample_count`, `sample_steps`,
and `sample_times_s`. Dust charge and force histories are saved separately in
`piel2017_dust_history.npz`.
