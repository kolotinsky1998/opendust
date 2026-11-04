#!/bin/bash
#SBATCH --job-name=od-figures
#SBATCH --account=proj_1827
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
echo "Starting postprocess on $(hostname) with $PYTHON_EXECUTABLE"
"$PYTHON_EXECUTABLE" -u "$EXAMPLE_DIR/postprocess.py" --config "$EXAMPLE_DIR/article_config.json"
