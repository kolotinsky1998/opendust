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
