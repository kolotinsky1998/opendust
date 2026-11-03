#!/bin/bash
#SBATCH --job-name=odj-fig8
#SBATCH --account=proj_1827
#SBATCH --partition=rocky
#SBATCH --time=1-00:00:00
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --array=0-11
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail

REPO_ROOT="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
SCRIPT_DIR="$REPO_ROOT/examples/OpenDust2023/Figure8"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

python -c "import jax, matplotlib; assert jax.default_backend() == 'gpu', jax.default_backend(); from opendust_jax import cuda_pair_forces_available; assert cuda_pair_forces_available(), 'CUDA FFI library is not built'"

srun python -u "$SCRIPT_DIR/run.py" --case-index "$SLURM_ARRAY_TASK_ID"
