#!/usr/bin/env python3
"""Build the optional JAX FFI CUDA pair-force shared library with nvcc."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import jax

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE = REPO_ROOT / "src/opendust_jax/cuda/pair_forces_cuda.cu"
DEFAULT_OUTPUT = (
    REPO_ROOT / "src/opendust_jax/_cuda/libopendust_jax_pair_forces.so"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cuda-arch",
        default="70",
        help="CUDA SM number without punctuation (70 for V100, 80 for A100).",
    )
    parser.add_argument("--nvcc", default="nvcc")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--no-fast-math",
        action="store_true",
        help="Use precise CUDA math instead of --use_fast_math.",
    )
    args = parser.parse_args()

    nvcc = shutil.which(args.nvcc)
    if nvcc is None:
        raise SystemExit(
            f"nvcc executable {args.nvcc!r} was not found. Load a CUDA toolkit "
            "module or install the matching cuda-nvcc package."
        )
    if not args.cuda_arch.isdigit():
        parser.error("--cuda-arch must look like 70 or 80")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        nvcc,
        str(SOURCE),
        "-o",
        str(args.output),
        "-shared",
        "-Xcompiler=-fPIC",
        "-std=c++17",
        "-O3",
        "--expt-relaxed-constexpr",
        f"-I{jax.ffi.include_dir()}",
        f"-gencode=arch=compute_{args.cuda_arch},code=sm_{args.cuda_arch}",
    ]
    if not args.no_fast_math:
        command.append("--use_fast_math")
    print("Building:", " ".join(command), flush=True)
    subprocess.run(command, check=True)
    print(f"Built {args.output}")


if __name__ == "__main__":
    main()
