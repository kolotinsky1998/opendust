#!/bin/bash
#SBATCH --job-name=od-chain-plots
#SBATCH --account=proj_1827
#SBATCH --partition=rocky
#SBATCH --time=01:00:00
set -euo pipefail

EXAMPLE_DIR="${OPENDUST_EXAMPLE_DIR:-${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}}"
test -f "$EXAMPLE_DIR/postprocess.py"
PYTHON_EXECUTABLE="${OPENDUST_PYTHON:-python3}"
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Starting chain-grid postprocessing on $(hostname)"
"$PYTHON_EXECUTABLE" -u "$EXAMPLE_DIR/postprocess.py" \
  --config "$EXAMPLE_DIR/chain_config.json"
