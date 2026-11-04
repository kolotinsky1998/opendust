#!/bin/bash
#SBATCH --job-name=od-branch
#SBATCH --account=proj_1827
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
set -euo pipefail

if [[ $# -ne 1 || ( "$1" != "linear" && "$1" != "quadratic" ) ]]; then
  echo "usage: $0 linear|quadratic" >&2
  exit 2
fi
EXAMPLE_DIR="$(cd "$(dirname "$0")" && pwd)"
export JAX_ENABLE_X64=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
python3 "$EXAMPLE_DIR/workflow.py" --config "$EXAMPLE_DIR/article_config.json" --stage "$1" --case-index "$SLURM_ARRAY_TASK_ID"
