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
