#!/bin/bash
#SBATCH --job-name=od-constant
#SBATCH --account=proj_1827
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
set -euo pipefail

EXAMPLE_DIR="${OPENDUST_EXAMPLE_DIR:-${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}}"
test -f "$EXAMPLE_DIR/workflow.py"
PYTHON_EXECUTABLE="${OPENDUST_PYTHON:-python3}"
if type module >/dev/null 2>&1; then
  module load cuda/12.9
fi
export JAX_ENABLE_X64=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
echo "Starting constant case $SLURM_ARRAY_TASK_ID on $(hostname) with $PYTHON_EXECUTABLE"
"$PYTHON_EXECUTABLE" -c 'import jax; print("JAX devices:", jax.devices(), flush=True); assert jax.default_backend() == "gpu", f"JAX backend is {jax.default_backend()}, expected gpu"; from opendust_jax import cuda_pair_forces_available; assert cuda_pair_forces_available(), "OpenDust CUDA FFI is unavailable"'
"$PYTHON_EXECUTABLE" -u "$EXAMPLE_DIR/workflow.py" --config "$EXAMPLE_DIR/article_config.json" --stage constant --case-index "$SLURM_ARRAY_TASK_ID"
