#!/bin/bash
#SBATCH --job-name=od-figures
#SBATCH --account=proj_1827
#SBATCH --time=01:00:00
set -euo pipefail

EXAMPLE_DIR="$(cd "$(dirname "$0")" && pwd)"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
python3 "$EXAMPLE_DIR/postprocess.py" --config "$EXAMPLE_DIR/article_config.json"
