#!/bin/bash
#SBATCH --job-name=od-figure3
#SBATCH --account=proj_1827
#SBATCH --partition=rocky
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
set -euo pipefail

EXAMPLE_DIR="${OPENDUST_EXAMPLE_DIR:-${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}}"
test -f "$EXAMPLE_DIR/figure3_workflow.py"
PYTHON_EXECUTABLE="${OPENDUST_PYTHON:-python3}"
if type module >/dev/null 2>&1; then
  module load cuda/12.9
fi
export JAX_ENABLE_X64=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
echo "Starting Figure 3 reproduction on $(hostname) with $PYTHON_EXECUTABLE"
"$PYTHON_EXECUTABLE" -c 'import jax; print("JAX devices:", jax.devices(), flush=True); assert jax.default_backend() == "gpu", f"JAX backend is {jax.default_backend()}, expected gpu"'
"$PYTHON_EXECUTABLE" -u "$EXAMPLE_DIR/figure3_workflow.py" \
  --config "$EXAMPLE_DIR/article_config.json" \
  --output-dir "$EXAMPLE_DIR/results/figure3_table_I" \
  --profile all
