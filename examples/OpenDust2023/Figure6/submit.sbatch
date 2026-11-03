#!/bin/bash
#SBATCH --job-name=odj-fig6
#SBATCH --account=proj_1827
#SBATCH --partition=test
#SBATCH --time=00:10:00
#SBATCH --gpus=1
#SBATCH --array=0-1
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

python -c "import jax, matplotlib; assert jax.default_backend() == 'gpu', jax.default_backend(); from opendust_jax import cuda_pair_forces_available; assert cuda_pair_forces_available(), 'CUDA FFI library is not built'"

srun python -u "$SCRIPT_DIR/run.py" --case-index "$SLURM_ARRAY_TASK_ID"
