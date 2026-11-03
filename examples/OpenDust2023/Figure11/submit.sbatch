#!/bin/bash -l
#SBATCH --job-name=odj-fig11
#SBATCH --account=proj_1827
#SBATCH --partition=rocky
#SBATCH --time=1-00:00:00
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --array=0-6
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

module load cuda/12.9
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate opendust-jax

export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

python -c "import jax, matplotlib; assert jax.default_backend() == 'gpu', jax.default_backend(); from opendust_jax import cuda_pair_forces_available; assert cuda_pair_forces_available(), 'CUDA FFI library is not built'"

srun python -u "$SCRIPT_DIR/run.py" --case-index "$SLURM_ARRAY_TASK_ID"
