#!/bin/bash
#SBATCH --job-name=od-constant
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
set -euo pipefail

EXAMPLE_DIR="$(cd "$(dirname "$0")" && pwd)"
export JAX_ENABLE_X64=1
python3 "$EXAMPLE_DIR/workflow.py" --config "$EXAMPLE_DIR/article_config.json" --stage constant --case-index "$SLURM_ARRAY_TASK_ID"
