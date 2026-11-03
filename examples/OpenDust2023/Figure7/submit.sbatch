#!/bin/bash
#SBATCH --job-name=odj-fig7
#SBATCH --account=proj_1827
#SBATCH --partition=rocky
#SBATCH --time=1-00:00:00
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --output=slurm-%x-%A.out
#SBATCH --error=slurm-%x-%A.err

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

python -c "import jax, matplotlib; assert jax.default_backend() == 'gpu', jax.default_backend(); from opendust_jax import cuda_pair_forces_available; assert cuda_pair_forces_available(), 'CUDA FFI library is not built'"

srun python -u "$SCRIPT_DIR/run.py" --case-index 0
