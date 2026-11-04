#!/bin/bash
set -euo pipefail

EXAMPLE_DIR="$(cd "$(dirname "$0")" && pwd)"
constant_job="$(sbatch --parsable "$EXAMPLE_DIR/slurm_constant.sh")"
linear_job="$(sbatch --parsable --dependency="afterok:$constant_job" "$EXAMPLE_DIR/slurm_branch.sh" linear)"
quadratic_job="$(sbatch --parsable --dependency="afterok:$constant_job" "$EXAMPLE_DIR/slurm_branch.sh" quadratic)"
post_job="$(sbatch --parsable --dependency="afterok:$linear_job:$quadratic_job" "$EXAMPLE_DIR/slurm_postprocess.sh")"
echo "constant=$constant_job linear=$linear_job quadratic=$quadratic_job postprocess=$post_job"
